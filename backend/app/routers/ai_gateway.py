# backend/app/routers/ai_gateway.py

import logging
import re
import asyncio
from typing import Any, List, Optional, Dict
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, validator

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ===== INPUT/OUTPUT GUARDRAILS =====

class MathInputGuardrails:
    """
    Input guardrails to ensure only educational mathematical content is processed.
    """
    
    # Allowed mathematical topics
    ALLOWED_TOPICS = {
        'arithmetic', 'algebra', 'geometry', 'calculus', 'trigonometry',
        'statistics', 'probability', 'number_theory', 'linear_algebra',
        'differential_equations', 'discrete_math', 'set_theory'
    }
    
    # Prohibited content patterns
    PROHIBITED_PATTERNS = [
        r'(?i)\b(sex|sexual|porn|nude|naked)\b',
        r'(?i)\b(drugs|marijuana|cocaine|heroin)\b',
        r'(?i)\b(kill|murder|suicide|harm)\b',
        r'(?i)\b(hack|exploit|virus|malware)\b',
        r'(?i)\b(gambling|bet|casino)\b',
        r'(?i)\b(politics|political|election)\b',
        r'(?i)\b(religion|religious|god|allah|jesus)\b'
    ]
    
    # Mathematical indicators
    MATH_INDICATORS = [
        r'\d+[\+\-\*\/\^\=]\d+',  # Basic arithmetic
        r'\b(solve|calculate|find|compute|evaluate)\b',
        r'\b(equation|formula|function|derivative|integral)\b',
        r'\b(triangle|circle|square|rectangle|polygon)\b',
        r'\b(sine|cosine|tangent|logarithm|exponential)\b',
        r'[xy]\s*[\+\-\*\/\=]',  # Variable expressions
        r'\b(matrix|vector|determinant)\b',
        r'\b(limit|series|sequence|sum)\b'
    ]
    
    @classmethod
    def validate_input(cls, text: str) -> Dict[str, Any]:
        """
        Comprehensive input validation for mathematical content.
        
        Returns:
            Dict with 'is_valid', 'reason', 'topic', 'confidence'
        """
        if not text or len(text.strip()) < 2:
            return {
                'is_valid': False,
                'reason': 'Input too short',
                'topic': None,
                'confidence': 0.0
            }
        
        text_clean = text.strip().lower()
        
        # Check for prohibited content
        for pattern in cls.PROHIBITED_PATTERNS:
            if re.search(pattern, text_clean):
                return {
                    'is_valid': False,
                    'reason': 'Contains prohibited content',
                    'topic': None,
                    'confidence': 0.0
                }
        
        # Check for mathematical indicators
        math_score = 0
        detected_topic = 'general_math'
        
        for indicator in cls.MATH_INDICATORS:
            if re.search(indicator, text_clean):
                math_score += 1
        
        # Topic detection
        if re.search(r'\b(derivative|integral|limit|calculus)\b', text_clean):
            detected_topic = 'calculus'
        elif re.search(r'\b(triangle|circle|area|volume|geometry)\b', text_clean):
            detected_topic = 'geometry'
        elif re.search(r'\b(matrix|vector|linear)\b', text_clean):
            detected_topic = 'linear_algebra'
        elif re.search(r'\b(solve.*=|equation|algebra)\b', text_clean):
            detected_topic = 'algebra'
        elif re.search(r'[\d\+\-\*\/]', text_clean):
            detected_topic = 'arithmetic'
        
        # Calculate confidence
        confidence = min(1.0, math_score / 3.0)
        
        # Minimum threshold for mathematical content
        is_valid = math_score > 0 or any(word in text_clean for word in [
            'what is', 'calculate', 'solve', 'find', 'compute'
        ])
        
        return {
            'is_valid': is_valid,
            'reason': 'Valid mathematical content' if is_valid else 'Not mathematical content',
            'topic': detected_topic if is_valid else None,
            'confidence': confidence
        }

class MathOutputGuardrails:
    """
    Output guardrails to ensure responses are educational and accurate.
    """
    
    @classmethod
    def validate_output(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output response for educational quality.
        
        Returns:
            Dict with 'is_valid', 'quality_score', 'improvements'
        """
        improvements = []
        quality_score = 1.0
        
        answer = response.get('answer', '')
        steps = response.get('steps', [])
        
        # Check if answer exists
        if not answer or answer.strip() == '':
            quality_score *= 0.3
            improvements.append('Missing answer')
        
        # Check for step-by-step explanation
        if not steps or len(steps) < 1:
            quality_score *= 0.7
            improvements.append('Missing step-by-step explanation')
        
        # Check for educational value
        educational_indicators = [
            'step', 'because', 'therefore', 'first', 'next', 'finally',
            'explanation', 'reason', 'method', 'approach'
        ]
        
        combined_text = f"{answer} {' '.join(steps)}".lower()
        educational_score = sum(1 for indicator in educational_indicators 
                              if indicator in combined_text)
        
        if educational_score < 2:
            quality_score *= 0.8
            improvements.append('Could be more educational')
        
        # Check answer format
        if re.search(r'^\d+(\.\d+)?$', answer.strip()):
            quality_score *= 1.1  # Bonus for numeric answers
        
        return {
            'is_valid': quality_score > 0.5,
            'quality_score': min(1.0, quality_score),
            'improvements': improvements
        }

# ===== PYDANTIC MODELS =====

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    student_level: Optional[str] = Field("intermediate", regex="^(beginner|intermediate|advanced)$")
    require_steps: bool = Field(True, description="Whether to include step-by-step solution")
    context: Optional[str] = Field(None, max_length=500, description="Additional context")

    @validator("question")
    def validate_mathematical_content(cls, v: str) -> str:
        """Validate that the question is mathematical and educational."""
        validation = MathInputGuardrails.validate_input(v)
        
        if not validation['is_valid']:
            raise ValueError(f"Invalid input: {validation['reason']}")
        
        if validation['confidence'] < 0.3:
            raise ValueError("Question must be more clearly mathematical")
        
        return v.strip()

class AnswerResponse(BaseModel):
    answer: str
    steps: List[str] = []
    source: str
    score: float = Field(..., ge=0.0, le=1.0)
    problem_type: Optional[str] = None
    confidence: Optional[str] = None
    educational_quality: Optional[float] = None

class GatewayResponse(BaseModel):
    query: str
    routed_to: str
    results: List[AnswerResponse]
    fallback_used: bool = False
    processing_time: Optional[float] = None
    guardrails_passed: bool = True
    quality_metrics: Optional[Dict[str, Any]] = None

# ===== ROUTING LOGIC =====

@router.post("/solve", response_model=GatewayResponse)
async def solve_math_problem(request: QueryRequest) -> GatewayResponse:
    """
    Enhanced math problem solving with comprehensive guardrails.
    """
    start_time = datetime.now()
    logger.info(f"🔍 Processing math query: {request.question[:100]}...")
    
    try:
        # Step 1: Enhanced input validation
        validation = MathInputGuardrails.validate_input(request.question)
        if not validation['is_valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input validation failed: {validation['reason']}"
            )
        
        # Step 2: Route to orchestration layer
        from app.orchestration import solve_math_problem as orchestrate_solution
        
        solution_response = await orchestrate_solution(request.question)
        
        # Step 3: Apply output guardrails
        enhanced_results = []
        for result in solution_response.results:
            # Validate output quality
            output_validation = MathOutputGuardrails.validate_output({
                'answer': result.answer,
                'steps': result.steps
            })
            
            # Enhance with educational metrics
            enhanced_result = AnswerResponse(
                answer=result.answer,
                steps=result.steps,
                source=result.source,
                score=result.score,
                problem_type=validation.get('topic'),
                educational_quality=output_validation.get('quality_score')
            )
            
            enhanced_results.append(enhanced_result)
        
        # Step 4: Calculate processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Step 5: Prepare enhanced response
        enhanced_response = GatewayResponse(
            query=request.question,
            routed_to=solution_response.routed_to,
            results=enhanced_results,
            fallback_used=solution_response.fallback_used,
            processing_time=processing_time,
            guardrails_passed=True,
            quality_metrics={
                'input_confidence': validation.get('confidence'),
                'detected_topic': validation.get('topic'),
                'total_results': len(enhanced_results),
                'avg_quality': sum(r.educational_quality or 0 for r in enhanced_results) / len(enhanced_results) if enhanced_results else 0
            }
        )
        
        logger.info(f"✅ Math query processed successfully in {processing_time:.2f}s")
        return enhanced_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Math query processing failed")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Return error response with guardrails info
        return GatewayResponse(
            query=request.question,
            routed_to="error_handler",
            results=[AnswerResponse(
                answer=f"I apologize, but I encountered an error processing your mathematical question: {str(e)}",
                steps=["Error occurred during processing", "Please try rephrasing your question"],
                source="error_handler",
                score=0.0,
                educational_quality=0.0
            )],
            fallback_used=True,
            processing_time=processing_time,
            guardrails_passed=False
        )

@router.post("/validate", response_model=Dict[str, Any])
async def validate_math_input(question: str) -> Dict[str, Any]:
    """
    Endpoint to validate mathematical input without solving.
    """
    validation = MathInputGuardrails.validate_input(question)
    return {
        "input": question,
        "validation": validation,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/topics", response_model=Dict[str, List[str]])
async def get_supported_topics():
    """
    Get list of supported mathematical topics.
    """
    return {
        "supported_topics": list(MathInputGuardrails.ALLOWED_TOPICS),
        "examples": {
            "arithmetic": ["What is 2+2?", "Calculate 15 * 7"],
            "algebra": ["Solve x + 5 = 10", "Find x in 2x - 3 = 7"],
            "geometry": ["Area of circle with radius 5", "Volume of sphere"],
            "calculus": ["Derivative of x^2", "Integral of 2x dx"]
        }
    }

@router.get("/health")
async def gateway_health():
    """Enhanced health check with guardrails status."""
    return {
        "status": "healthy",
        "component": "Enhanced AI Gateway",
        "guardrails": {
            "input_validation": "active",
            "output_validation": "active",
            "content_filtering": "active"
        },
        "supported_routes": [
            "arithmetic_calculator",
            "knowledge_base",
            "chain_of_thought", 
            "web_search",
            "fallback"
        ],
        "timestamp": datetime.now().isoformat()
    }