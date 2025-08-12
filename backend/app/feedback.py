import logging
import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services.db_service import DBService

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackCreate(BaseModel):
    question_id:   str
    response_id:   str
    user_rating:   int               = Field(..., ge=1, le=5)
    feedback_text: Optional[str]     = None
    categories:    List[str]         = Field(default_factory=list)

    @validator("categories", each_item=True)
    def validate_category(cls, v: str) -> str:
        allowed = {"accuracy", "clarity", "completeness", "style", "other"}
        if v not in allowed:
            raise ValueError(f"Unknown category '{v}'")
        return v


class FeedbackRead(BaseModel):
    id:            str
    question_id:   str
    response_id:   str
    user_rating:   int
    feedback_text: Optional[str]
    categories:    List[str]
    timestamp:     datetime.datetime


@router.post(
    "",
    response_model=FeedbackRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_feedback(
    payload: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
) -> FeedbackRead:
    svc = DBService(db)
    try:
        fb = await svc.create_feedback(
            question_id=payload.question_id,
            response_id=payload.response_id,
            feedback_text=payload.feedback_text,
            user_rating=payload.user_rating,
        )
    except Exception:
        logger.exception("Failed to create feedback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save feedback",
        )

    return FeedbackRead(
        id=fb.id,
        question_id=fb.question_id,
        response_id=fb.response_id,
        user_rating=fb.rating or 0,
        feedback_text=fb.user_feedback,
        categories=payload.categories,
        timestamp=fb.timestamp,
    )


@router.get(
    "",
    response_model=List[FeedbackRead],
    status_code=status.HTTP_200_OK,
)
async def list_feedback(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> List[FeedbackRead]:
    svc = DBService(db)
    rows = await svc.list_feedback(limit=limit)
    return [
        FeedbackRead(
            id=fb.id,
            question_id=fb.question_id,
            response_id=fb.response_id,
            user_rating=fb.rating or 0,
            feedback_text=fb.user_feedback,
            categories=[],  # categories aren’t stored
            timestamp=fb.timestamp,
        )
        for fb in rows
    ]
