import logging
import re
import httpx
import sympy as sp
import asyncio
from typing import Any, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator

from app.embedding import get_embedding
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Pydantic Models ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

    @validator("question")
    def must_be_math(cls, v: str) -> str:
        # ✅ Use same logic as search_validator.py for consistency
        math_pattern = re.compile(r'[\d\+\-\*\/\=\^\(\)]')
        math_keywords = ["integral", "derivative", "solve", "limit", "equation", "calculate", "what is", "find"]
        
        if not (math_pattern.search(v) or any(k in v.lower() for k in math_keywords)):
            raise ValueError("Question must be math-related")
        if re.search(r"(?i)(sex|drugs|kill)", v):
            raise ValueError("Inappropriate content detected")
        return v.strip()


class AnswerResponse(BaseModel):
    answer: str
    steps: Optional[List[str]] = None
    source: str
    score: float


class GatewayResponse(BaseModel):
    query: str
    routed_to: str
    results: List[AnswerResponse]
    fallback_used: bool = False


# --- Helper Guardrail Functions ---

def sympy_validate(expr: str) -> bool:
    """Validate if expression can be parsed by SymPy"""
    if not expr or not isinstance(expr, str):
        return False
    try:
        # Clean the expression first
        cleaned = expr.strip()
        if not cleaned:
            return False
        sp.sympify(cleaned)
        return True
    except (sp.SympifyError, TypeError, ValueError):
        return False


async def call_gemini(prompt: str) -> str:
    """Call Gemini API with proper error handling"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={settings.GOOGLE_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
            }
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise


# --- Routing Logic ---

@router.post("/query", response_model=GatewayResponse)
async def query_math(request: QueryRequest) -> GatewayResponse:
    q = request.question
    logger.info("Received query: %s", q)

    try:
        # 1. Embed question
        embedding: List[float] = await get_embedding(q)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        # Continue without embedding-based search
        embedding = None

    # 2. Query KB if embedding succeeded
    if embedding:
        try:
            from app.lancedb_store import LanceDBStore
            kb_store = LanceDBStore(db_path=settings.LANCEDB_PATH, table_name="math_qa")
            kb_hits = await kb_store.semantic_search(embedding, top_k=3, threshold=0.65)
            
            if kb_hits:
                results = []
                for hit in kb_hits:
                    ans = hit.get("answer", "")
                    if ans and sympy_validate(ans):
                        steps = hit.get("steps")
                        results.append(AnswerResponse(
                            answer=ans,
                            steps=steps if isinstance(steps, list) else [steps] if steps else None,
                            source="knowledge_base",
                            score=float(hit.get("score", 0.0))
                        ))
                    else:
                        logger.warning("KB hit failed SymPy validation: %s", ans)
                
                if results:
                    return GatewayResponse(
                        query=q,
                        routed_to="knowledge_base",
                        results=results,
                        fallback_used=False
                    )
        except Exception as e:
            logger.exception(f"Knowledge base search failed: {e}")

    # 3. Fallback → web search
    try:
        from app.search_validator import generate_search_response
        validated = generate_search_response(q)
        
        if validated and validated.get("steps"):
            steps = validated["steps"]
            answer = validated.get("answer", steps[-1] if steps else "No answer found")
            
            return GatewayResponse(
                query=q,
                routed_to="web_search",
                results=[AnswerResponse(
                    answer=answer,
                    steps=steps,
                    source="web_search",
                    score=0.5
                )],
                fallback_used=True
            )
    except Exception as e:
        logger.exception(f"Web search fallback failed: {e}")

    # 4. Fallback → chain-of-thought
    try:
        from app.dspy_integration import MathCoTPipeline
        cot_pipeline = MathCoTPipeline(api_key=settings.GOOGLE_API_KEY)
        cot_out = await asyncio.to_thread(cot_pipeline.solve, q)
        
        if cot_out and isinstance(cot_out, dict):
            steps = cot_out.get("steps", [])
            answer = cot_out.get("answer", "")
            
            # More lenient validation for CoT
            if answer and (sympy_validate(answer) or len(answer.strip()) > 0):
                return GatewayResponse(
                    query=q,
                    routed_to="chain_of_thought",
                    results=[AnswerResponse(
                        answer=answer,
                        steps=steps if isinstance(steps, list) else [steps] if steps else None,
                        source="chain_of_thought",
                        score=0.3
                    )],
                    fallback_used=True
                )
    except Exception as e:
        logger.exception(f"CoT fallback failed: {e}")

    # 5. Final fallback - simple Gemini call
    try:
        prompt = f"Solve this math problem step by step: {q}\nProvide a clear final answer."
        gemini_response = await call_gemini(prompt)
        
        if gemini_response:
            return GatewayResponse(
                query=q,
                routed_to="gemini_direct",
                results=[AnswerResponse(
                    answer=gemini_response,
                    steps=[gemini_response],
                    source="gemini_direct",
                    score=0.1
                )],
                fallback_used=True
            )
    except Exception as e:
        logger.exception(f"Direct Gemini fallback failed: {e}")

    # If all routes failed
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="All routing methods failed. Please try again later."
    )


# --- Additional health check for the gateway ---
@router.get("/gateway/health")
async def gateway_health():
    """Check if the AI gateway is responding"""
    return {
        "status": "healthy",
        "component": "AI Gateway",
        "routes_available": [
            "knowledge_base",
            "web_search", 
            "chain_of_thought",
            "gemini_direct"
        ]
    }