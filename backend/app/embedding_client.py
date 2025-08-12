# backend/app/embedding_client.py

from typing import List
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> List[float]:
    """
    Convert input text to an embedding vector using SentenceTransformer.
    """
    return model.encode(text).tolist()
