# ===== app/models/models.py =====
import datetime
import uuid
import logging

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON, Index
from sqlalchemy.dialects.sqlite import BLOB

from app.db import Base  # ← Shared Base

logger = logging.getLogger(__name__)


def gen_uuid() -> str:
    return str(uuid.uuid4())


class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(String, primary_key=True, default=gen_uuid)
    question_id = Column(String, nullable=False, index=True)      # ← ADDED
    response_id = Column(String, nullable=False, index=True)      # ← ADDED
    question = Column(Text, nullable=False)                       # ← KEPT EXISTING
    generated_answer = Column(Text, nullable=False)               # ← KEPT EXISTING
    user_feedback = Column(Text, nullable=True)
    rating = Column(Integer, nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_feedback_rating_timestamp", "rating", "timestamp"),
        Index("ix_feedback_question_id", "question_id"),             # ← ADDED
        Index("ix_feedback_response_id", "response_id"),             # ← ADDED
    )


class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(String, primary_key=True, default=gen_uuid)
    question = Column(Text, nullable=False, index=True)
    source = Column(String, nullable=False)
    answer = Column(Text, nullable=False)
    response_time = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_history_question_timestamp", "question", "timestamp"),
    )


class KnowledgeBaseMetadata(Base):
    __tablename__ = "kb_metadata"
    
    id = Column(String, primary_key=True, default=gen_uuid)
    doc_id = Column(String, nullable=False, unique=True, index=True)
    title = Column(String, nullable=True)
    embedding = Column(BLOB, nullable=False)
    metadata_json = Column("metadata", JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_kb_timestamp", "timestamp"),
    )