# backend/app/embedding.py

from typing import List
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# load once on import
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Loaded embedding model all-MiniLM-L6-v2")

async def get_embedding(text: str) -> List[float]:
    """
    Async wrapper around SentenceTransformer.encode.
    Returns a single embedding vector for the input text.
    """
    # running encode in a thread to avoid blocking the event loop
    from asyncio import to_thread
    try:
        vec = await to_thread(_model.encode, text, convert_to_numpy=True)
        return vec.tolist()
    except Exception:
        logger.exception("Failed to generate embedding")
        raise
