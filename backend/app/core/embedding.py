# backend/app/embedding.py

import logging
import asyncio
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Enhanced embedding service with caching and error handling.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dimension = settings.EMBEDDING_DIMENSION
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the embedding model."""
        if self._model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"✅ Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise
        return self._model
    
    @lru_cache(maxsize=1000)
    def _cached_encode(self, text: str) -> tuple:
        """Cached encoding to avoid recomputing embeddings for same text."""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return tuple(embedding.tolist())
        except Exception as e:
            logger.error(f"❌ Encoding failed for text: {text[:50]}... Error: {e}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Clean and normalize text
        cleaned_text = self._preprocess_text(text)
        
        try:
            # Use asyncio.to_thread for true async execution
            embedding_tuple = await asyncio.to_thread(self._cached_encode, cleaned_text)
            return list(embedding_tuple)
        except Exception as e:
            logger.exception(f"❌ Failed to generate embedding for: {text[:50]}...")
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Preprocess batch
                cleaned_batch = [self._preprocess_text(text) for text in batch]
                
                # Generate embeddings for batch
                batch_embeddings = await asyncio.to_thread(
                    self.model.encode, 
                    cleaned_batch,
                    convert_to_numpy=True
                )
                
                # Convert to list of lists
                embeddings.extend([emb.tolist() for emb in batch_embeddings])
                
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"❌ Batch embedding failed for batch starting at {i}: {e}")
                # Add zero embeddings for failed batch
                embeddings.extend([[0.0] * self._dimension for _ in batch])
        
        return embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Handle mathematical notation
        cleaned = cleaned.replace('×', '*')
        cleaned = cleaned.replace('÷', '/')
        cleaned = cleaned.replace('²', '^2')
        cleaned = cleaned.replace('³', '^3')
        
        # Limit length to prevent issues
        max_length = 512
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        return cleaned
    
    def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "dimension": self._dimension,
            "is_loaded": self._model is not None,
            "cache_size": self._cached_encode.cache_info()._asdict() if hasattr(self._cached_encode, 'cache_info') else {}
        }

# Global embedding service instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

# Backward compatibility functions
async def get_embedding(text: str) -> List[float]:
    """
    Backward compatible function for getting embeddings.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of float values representing the embedding
    """
    service = get_embedding_service()
    return await service.get_embedding(text)

async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Backward compatible function for batch embeddings.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of embeddings
    """