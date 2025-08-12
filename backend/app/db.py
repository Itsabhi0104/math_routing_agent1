# backend/app/db.py
from typing import AsyncGenerator
import logging

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Database URL and engine
# ----------------------------------------------------
DATABASE_URL = "sqlite+aiosqlite:///./app.db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

# ----------------------------------------------------
# Shared Base for all ORM models
# ----------------------------------------------------
Base = declarative_base()

# ----------------------------------------------------
# Async session factory
# ----------------------------------------------------
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ----------------------------------------------------
# FastAPI dependency to get a DB session
# ----------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
