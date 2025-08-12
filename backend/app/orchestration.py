# backend/app/orchestration.py

import logging
import os
import re
from typing import List, Dict, Any, Optional

from app.search_validator import validate_math_question, generate_search_response
from app.lancedb_store import LanceDBStore
from app.embedding_client import get_embedding
from app.ai_gateway import GatewayResponse, AnswerResponse
from app.utils_math import evaluate_simple_expression

logger = logging.getLogger(__name__)

# Configurable KB threshold via environment variable (default 0.75)
KB_THRESHOLD = float(os.getenv("KB_THRESHOLD", "0.75"))

# Initialize LanceDB store once (adjust path/name as needed)
store = LanceDBStore(
    db_path=os.getenv("LANCEDB_PATH", "lancedb_math"),
    table_name=os.getenv("LANCEDB_TABLE", "math_qa"),
    vector_column=os.getenv("LANCEDB_VECTOR_COL", "embedding"),
)

def detect_two_points_center_on_line_contradiction(text: str) -> Optional[str]:
    """
    Heuristic to detect the pattern:
      "passing through (x1,y1) and (x2,y2) ... center lies on the line x+y=C"
    If it finds that the two constraints are inconsistent (no real center), return a message.
    """
    m = re.search(r"passing through \(([^)]+)\) and \(([^)]+)\).*center lies on the line x\+y=([0-9\.\-]+)", text)
    if not m:
        return None
    try:
        p1 = [float(x) for x in m.group(1).split(",")]
        p2 = [float(x) for x in m.group(2).split(",")]
        c = float(m.group(3))
        # For now we don't claim contradiction unless proven; return None
        return None
    except Exception:
        return None


async def _map_items_to_answer_responses(items: List[Dict[str, Any]]) -> List[AnswerResponse]:
    results: List[AnswerResponse] = []
    for item in items:
        answer = item.get("answer", "")
        steps = item.get("steps", []) or []
        score = float(item.get("score", 0.0))
        source = item.get("source", "web_search")
        results.append(
            AnswerResponse(
                answer=answer,
                steps=steps,
                source=source,
                score=score,
            )
        )
    return results

async def solve_math_problem(question: str) -> GatewayResponse:
    """
    Flow:
      - Quick arithmetic short-circuit (always first)
      - logic_check
      - validation and KB exact/semantic search / web fallback
    """
    logger.info("Received question: %s", question)

    # 0) Arithmetic short-circuit FIRST (so punctuation/validator won't block)
    try:
        ar_eval = evaluate_simple_expression(question)
    except Exception:
        ar_eval = None

    if ar_eval is not None:
        rendered, numeric = ar_eval
        logger.info("Arithmetic short-circuit: %s -> %s", question, rendered)
        return GatewayResponse(
            query=question,
            routed_to="calculator",
            source="calculator",
            fallback_used=False,
            results=[
                AnswerResponse(answer=rendered, steps=["Computed locally using SymPy"], source="calculator", score=1.0)
            ],
        )

    # 1) Quick deterministic logic check for special contradictions
    contradiction_msg = detect_two_points_center_on_line_contradiction(question)
    if contradiction_msg:
        logger.info("Logic-check detected contradiction for question: %s", question)
        return GatewayResponse(query=question, routed_to="logic_check", results=[AnswerResponse(answer=contradiction_msg, steps=[], source="logic_check", score=1.0)], fallback_used=False)

    # 2) Ensure question is math-related
    try:
        is_math = validate_math_question(question)
    except Exception:
        is_math = False

    if not is_math:
        logger.info("Question not math-related; routing to web search: %s", question)
        web_items = generate_search_response(question)
        results = await _map_items_to_answer_responses(web_items)
        return GatewayResponse(query=question, routed_to="web_search", source="web_search", fallback_used=True, results=results)

    # 3) Exact-match check in the KB before semantic search
    try:
        exact = store.find_exact_match(question)
        if exact:
            logger.info("Exact KB match found for question")
            return GatewayResponse(
                query=question,
                routed_to="knowledge_base",
                source=exact.get("source", "lance_db_exact"),
                fallback_used=False,
                results=[AnswerResponse(answer=exact.get("answer", ""), steps=[], source=exact.get("source", "lance_db_exact"), score=exact.get("score", 1.0))],
            )
    except Exception:
        logger.exception("Exact-match lookup failed")

    # 4) Compute embedding and do semantic search
    try:
        embedding: List[float] = get_embedding(question)
    except Exception as e:
        logger.exception("Embedding failed, falling back to web search: %s", e)
        web_items = generate_search_response(question)
        results = await _map_items_to_answer_responses(web_items)
        return GatewayResponse(
            query=question,
            routed_to="web_search",
            source="web_search",
            fallback_used=True,
            results=results,
        )

    hits = await store.semantic_search(
        query_embedding=embedding,
        top_k=5,
        threshold=KB_THRESHOLD,
    )

    if hits:
        results = []
        for item in hits:
            results.append(
                AnswerResponse(
                    answer=item.get("answer", ""),
                    steps=item.get("steps") or [],
                    source=item.get("source", "lance_db"),
                    score=float(item.get("score", 0.0)),
                )
            )
        return GatewayResponse(
            query=question,
            routed_to="knowledge_base",
            source="lance_db",
            fallback_used=False,
            results=results,
        )

    # No KB hits above threshold -> fallback to web search
    logger.info("No KB hits ≥ threshold (=%s) for: %s — using web search fallback", KB_THRESHOLD, question)
    web_items = generate_search_response(question)
    results = await _map_items_to_answer_responses(web_items)
    return GatewayResponse(
        query=question,
        routed_to="web_search",
        source="web_search",
        fallback_used=True,
        results=results,
    )
