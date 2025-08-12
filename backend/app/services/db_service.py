import datetime
import logging
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    UserFeedback,
    QueryHistory,
    KnowledgeBaseMetadata,
)

logger = logging.getLogger(__name__)


class DBService:
    """
    Provides CRUD operations for feedback, history and KB metadata.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # --- UserFeedback CRUD ---

    async def create_feedback(
        self,
        question_id: str,
        response_id: str,
        feedback_text: Optional[str] = None,
        user_rating: Optional[int] = None,
    ) -> UserFeedback:
        """
        Store a new feedback record, tied to a history entry.
        """
        # Get the original question and answer from QueryHistory
        original_question = ""
        original_answer = ""
        
        try:
            # Try to get the original question from QueryHistory using question_id
            history_result = await self.session.execute(
                select(QueryHistory).where(QueryHistory.id == question_id)
            )
            history_record = history_result.scalars().first()
            
            if history_record:
                original_question = history_record.question
                original_answer = history_record.answer
            else:
                # Fallback: use question_id as the question text
                original_question = f"Question ID: {question_id}"
                original_answer = f"Response ID: {response_id}"
        except Exception as e:
            logger.warning(f"Could not fetch original question/answer: {e}")
            original_question = f"Question ID: {question_id}"
            original_answer = f"Response ID: {response_id}"

        fb = UserFeedback(
            question_id=question_id,                    # ← NOW WORKS
            response_id=response_id,                    # ← NOW WORKS
            question=original_question,                 # ← POPULATE FROM HISTORY
            generated_answer=original_answer,           # ← POPULATE FROM HISTORY
            user_feedback=feedback_text,
            rating=user_rating,
            timestamp=datetime.datetime.utcnow(),
        )
        
        try:
            self.session.add(fb)
            await self.session.commit()
            await self.session.refresh(fb)
            logger.debug("Created UserFeedback id=%s", fb.id)
            return fb
        except Exception:
            logger.exception("Failed to create feedback")
            await self.session.rollback()
            raise

    async def list_feedback(self, limit: int = 100) -> List[UserFeedback]:
        result = await self.session.execute(
            select(UserFeedback).order_by(UserFeedback.timestamp.desc()).limit(limit)
        )
        return result.scalars().all()

    # --- QueryHistory CRUD ---

    async def log_query(
        self,
        question: str,
        source: str,
        answer: str,
        response_time: float,
    ) -> QueryHistory:
        history = QueryHistory(
            question=question,
            source=source,
            answer=answer,
            response_time=response_time,
            timestamp=datetime.datetime.utcnow(),
        )
        try:
            self.session.add(history)
            await self.session.commit()
            await self.session.refresh(history)
            logger.debug("Logged QueryHistory id=%s", history.id)
            return history
        except Exception:
            logger.exception("Failed to log query")
            await self.session.rollback()
            raise

    async def list_history(self, limit: int = 100) -> List[dict]:
        result = await self.session.execute(
            select(QueryHistory)
            .order_by(QueryHistory.timestamp.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "question_id": row.id,
                "question": row.question,
                "source": row.source,
                "answer": row.answer,
                "response_time": row.response_time,
                "timestamp": row.timestamp.isoformat(),
            }
            for row in rows
        ]

    # --- KnowledgeBaseMetadata CRUD ---

    async def add_kb_entry(
        self,
        doc_id: str,
        embedding: bytes,
        metadata: dict,
        title: Optional[str] = None,
    ) -> KnowledgeBaseMetadata:
        entry = KnowledgeBaseMetadata(
            doc_id=doc_id,
            embedding=embedding,
            metadata_json=metadata,
            title=title,
            timestamp=datetime.datetime.utcnow(),
        )
        try:
            self.session.add(entry)
            await self.session.commit()
            await self.session.refresh(entry)
            logger.debug("Added KB entry id=%s", entry.id)
            return entry
        except Exception:
            logger.exception("Failed to add KB entry")
            await self.session.rollback()
            raise

    async def get_kb_entry(self, doc_id: str) -> Optional[KnowledgeBaseMetadata]:
        result = await self.session.execute(
            select(KnowledgeBaseMetadata).where(
                KnowledgeBaseMetadata.doc_id == doc_id
            )
        )
        return result.scalars().first()