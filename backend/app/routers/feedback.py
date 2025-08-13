# backend/app/feedback.py

import logging
import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services.db_service import DBService

logger = logging.getLogger(__name__)
router = APIRouter()

# ===== PYDANTIC MODELS =====

class FeedbackCreate(BaseModel):
    """Model for creating new feedback."""
    question_id: str = Field(..., description="ID of the question that was answered")
    response_id: str = Field(..., description="ID of the response that was generated")
    user_rating: int = Field(..., ge=1, le=5, description="User rating from 1 (poor) to 5 (excellent)")
    feedback_text: Optional[str] = Field(None, max_length=1000, description="Optional text feedback")
    categories: List[str] = Field(default_factory=list, description="Feedback categories")
    improvement_suggestions: Optional[str] = Field(None, max_length=500, description="Suggestions for improvement")
    
    @validator("categories", each_item=True)
    def validate_category(cls, v: str) -> str:
        """Validate feedback categories."""
        allowed = {
            "accuracy", "clarity", "completeness", "speed", "style", 
            "educational_value", "step_by_step", "explanation", "other"
        }
        if v not in allowed:
            raise ValueError(f"Category '{v}' not allowed. Use one of: {allowed}")
        return v
    
    @validator("feedback_text")
    def validate_feedback_text(cls, v: Optional[str]) -> Optional[str]:
        """Validate feedback text content."""
        if v is None:
            return v
        
        # Remove excessive whitespace
        cleaned = v.strip()
        if not cleaned:
            return None
            
        # Basic content filtering (can be enhanced)
        prohibited_words = ["spam", "advertisement", "buy now"]
        if any(word in cleaned.lower() for word in prohibited_words):
            raise ValueError("Feedback contains prohibited content")
        
        return cleaned

class FeedbackResponse(BaseModel):
    """Model for feedback responses."""
    id: str
    question_id: str
    response_id: str
    user_rating: int
    feedback_text: Optional[str]
    categories: List[str]
    improvement_suggestions: Optional[str]
    timestamp: datetime.datetime
    processed: bool = False
    helpful_count: int = 0

class FeedbackSummary(BaseModel):
    """Model for feedback summary statistics."""
    total_feedback: int
    average_rating: float
    rating_distribution: Dict[int, int]
    top_categories: List[Dict[str, Any]]
    recent_feedback_count: int
    improvement_trends: Dict[str, Any]

class FeedbackAnalytics(BaseModel):
    """Model for detailed feedback analytics."""
    summary: FeedbackSummary
    quality_metrics: Dict[str, float]
    common_issues: List[str]
    positive_highlights: List[str]
    recommendations: List[str]

# ===== FEEDBACK ENDPOINTS =====

@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def create_feedback(
    payload: FeedbackCreate,
    db: AsyncSession = Depends(get_db)
) -> FeedbackResponse:
    """
    Create new user feedback for a math problem response.
    
    This endpoint allows users to provide ratings and comments on
    the quality and usefulness of generated mathematical solutions.
    """
    logger.info(f"📝 Creating feedback for question_id: {payload.question_id}")
    
    try:
        svc = DBService(db)
        
        # Create feedback record
        feedback_record = await svc.create_feedback(
            question_id=payload.question_id,
            response_id=payload.response_id,
            feedback_text=payload.feedback_text,
            user_rating=payload.user_rating
        )
        
        # Log feedback for analytics
        logger.info(f"✅ Feedback created: rating={payload.user_rating}, categories={payload.categories}")
        
        # Trigger adaptive learning if enabled
        try:
            await _trigger_adaptive_learning(feedback_record, payload)
        except Exception as e:
            logger.warning(f"⚠️ Adaptive learning trigger failed: {e}")
        
        return FeedbackResponse(
            id=feedback_record.id,
            question_id=feedback_record.question_id,
            response_id=feedback_record.response_id,
            user_rating=feedback_record.rating or 0,
            feedback_text=feedback_record.user_feedback,
            categories=payload.categories,
            improvement_suggestions=payload.improvement_suggestions,
            timestamp=feedback_record.timestamp,
            processed=False,
            helpful_count=0
        )
        
    except Exception as e:
        logger.exception("❌ Failed to create feedback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save feedback: {str(e)}"
        )

@router.get("", response_model=List[FeedbackResponse])
async def list_feedback(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of feedback items to return"),
    min_rating: Optional[int] = Query(None, ge=1, le=5, description="Minimum rating filter"),
    category: Optional[str] = Query(None, description="Filter by category"),
    recent_days: Optional[int] = Query(None, ge=1, le=365, description="Filter by recent days"),
    db: AsyncSession = Depends(get_db)
) -> List[FeedbackResponse]:
    """
    Retrieve feedback records with optional filtering.
    
    Supports filtering by rating, category, and time range for analysis.
    """
    logger.info(f"📊 Listing feedback: limit={limit}, min_rating={min_rating}, category={category}")
    
    try:
        svc = DBService(db)
        
        # Get feedback records
        feedback_records = await svc.list_feedback(limit=limit)
        
        # Convert to response models with filtering
        feedback_responses = []
        cutoff_date = None
        
        if recent_days:
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=recent_days)
        
        for record in feedback_records:
            # Apply filters
            if min_rating and (record.rating or 0) < min_rating:
                continue
            
            if cutoff_date and record.timestamp < cutoff_date:
                continue
            
            # Create response object
            response = FeedbackResponse(
                id=record.id,
                question_id=record.question_id,
                response_id=record.response_id,
                user_rating=record.rating or 0,
                feedback_text=record.user_feedback,
                categories=[],  # Categories not stored in current model
                improvement_suggestions=None,
                timestamp=record.timestamp,
                processed=True  # Assume processed if retrieved
            )
            
            feedback_responses.append(response)
        
        logger.info(f"📋 Returning {len(feedback_responses)} feedback items")
        return feedback_responses
        
    except Exception as e:
        logger.exception("❌ Failed to list feedback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve feedback: {str(e)}"
        )

@router.get("/analytics", response_model=FeedbackAnalytics)
async def get_feedback_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db)
) -> FeedbackAnalytics:
    """
    Get comprehensive feedback analytics and insights.
    
    Provides detailed analysis of user feedback trends, common issues,
    and recommendations for system improvement.
    """
    logger.info(f"📈 Generating feedback analytics for {days} days")
    
    try:
        svc = DBService(db)
        
        # Get recent feedback
        feedback_records = await svc.list_feedback(limit=1000)
        
        # Filter by date range
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        recent_feedback = [
            f for f in feedback_records 
            if f.timestamp >= cutoff_date and f.rating is not None
        ]
        
        # Calculate summary statistics
        if recent_feedback:
            ratings = [f.rating for f in recent_feedback]
            avg_rating = sum(ratings) / len(ratings)
            
            # Rating distribution
            rating_dist = {}
            for rating in range(1, 6):
                rating_dist[rating] = len([r for r in ratings if r == rating])
        else:
            avg_rating = 0.0
            rating_dist = {i: 0 for i in range(1, 6)}
        
        # Analyze feedback text for common themes
        feedback_texts = [f.user_feedback for f in recent_feedback if f.user_feedback]
        common_issues, positive_highlights = _analyze_feedback_text(feedback_texts)
        
        # Generate recommendations
        recommendations = _generate_recommendations(recent_feedback, avg_rating)
        
        # Build analytics response
        summary = FeedbackSummary(
            total_feedback=len(recent_feedback),
            average_rating=round(avg_rating, 2),
            rating_distribution=rating_dist,
            top_categories=[],  # Would need category storage
            recent_feedback_count=len(recent_feedback),
            improvement_trends={}
        )
        
        quality_metrics = {
            "user_satisfaction": avg_rating / 5.0,
            "response_rate": min(1.0, len(recent_feedback) / max(1, days * 10)),  # Estimate
            "positive_feedback_ratio": len([r for r in ratings if r >= 4]) / max(1, len(ratings))
        }
        
        analytics = FeedbackAnalytics(
            summary=summary,
            quality_metrics=quality_metrics,
            common_issues=common_issues,
            positive_highlights=positive_highlights,
            recommendations=recommendations
        )
        
        logger.info(f"📊 Analytics generated: avg_rating={avg_rating:.2f}, total_feedback={len(recent_feedback)}")
        return analytics
        
    except Exception as e:
        logger.exception("❌ Failed to generate analytics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not generate analytics: {str(e)}"
        )

@router.get("/summary", response_model=FeedbackSummary)
async def get_feedback_summary(
    db: AsyncSession = Depends(get_db)
) -> FeedbackSummary:
    """Get a quick summary of recent feedback."""
    try:
        svc = DBService(db)
        feedback_records = await svc.list_feedback(limit=100)
        
        # Calculate basic metrics
        if feedback_records:
            ratings = [f.rating for f in feedback_records if f.rating]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
            
            rating_dist = {}
            for rating in range(1, 6):
                rating_dist[rating] = len([r for r in ratings if r == rating])
        else:
            avg_rating = 0.0
            rating_dist = {i: 0 for i in range(1, 6)}
        
        return FeedbackSummary(
            total_feedback=len(feedback_records),
            average_rating=round(avg_rating, 2),
            rating_distribution=rating_dist,
            top_categories=[],
            recent_feedback_count=len(feedback_records),
            improvement_trends={}
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get feedback summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve feedback summary"
        )

# ===== HELPER FUNCTIONS =====

async def _trigger_adaptive_learning(feedback_record, feedback_payload: FeedbackCreate):
    """Trigger adaptive learning based on feedback."""
    try:
        # Only trigger for high-quality feedback
        if feedback_record.rating >= 4:
            logger.info(f"🎯 Triggering adaptive learning for high-rated feedback")
            
            # Here you would integrate with your adaptive learning system
            # For example, updating embeddings, refining prompts, etc.
            
            # Placeholder for actual implementation
            pass
            
    except Exception as e:
        logger.error(f"❌ Adaptive learning trigger failed: {e}")

def _analyze_feedback_text(feedback_texts: List[str]) -> tuple[List[str], List[str]]:
    """Analyze feedback text to extract common issues and positive highlights."""
    if not feedback_texts:
        return [], []
    
    # Simple keyword analysis (can be enhanced with NLP)
    issue_keywords = {
        'slow': 'response time',
        'wrong': 'accuracy',
        'confusing': 'clarity',
        'incomplete': 'completeness',
        'difficult': 'complexity'
    }
    
    positive_keywords = {
        'excellent': 'overall quality',
        'clear': 'clarity',
        'helpful': 'usefulness',
        'accurate': 'accuracy',
        'fast': 'speed'
    }
    
    common_issues = []
    positive_highlights = []
    
    # Count keyword occurrences
    issue_counts = {}
    positive_counts = {}
    
    for text in feedback_texts:
        text_lower = text.lower()
        
        for keyword, category in issue_keywords.items():
            if keyword in text_lower:
                issue_counts[category] = issue_counts.get(category, 0) + 1
        
        for keyword, category in positive_keywords.items():
            if keyword in text_lower:
                positive_counts[category] = positive_counts.get(category, 0) + 1
    
    # Get top issues and highlights
    common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    common_issues = [f"{category} ({count} mentions)" for category, count in common_issues]
    
    positive_highlights = sorted(positive_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    positive_highlights = [f"{category} ({count} mentions)" for category, count in positive_highlights]
    
    return common_issues, positive_highlights

def _generate_recommendations(feedback_records: List, avg_rating: float) -> List[str]:
    """Generate improvement recommendations based on feedback analysis."""
    recommendations = []
    
    if avg_rating < 3.0:
        recommendations.append("Critical: Overall satisfaction is low. Review response quality and accuracy.")
    elif avg_rating < 4.0:
        recommendations.append("Moderate: Response quality needs improvement. Focus on clarity and completeness.")
    else:
        recommendations.append("Good: Maintain current quality standards.")
    
    # Rating-based recommendations
    low_ratings = [f for f in feedback_records if f.rating and f.rating <= 2]
    if len(low_ratings) > len(feedback_records) * 0.2:  # More than 20% low ratings
        recommendations.append("High priority: Address accuracy and clarity issues.")
    
    # Volume-based recommendations
    if len(feedback_records) < 10:
        recommendations.append("Encourage more user feedback to improve system learning.")
    
    return recommendations