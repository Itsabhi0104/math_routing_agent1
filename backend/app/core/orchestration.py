# backend/app/orchestration.py - FIXED VERSION

import logging
import os
import re
from typing import List, Dict, Any, Optional
import sympy as sp
from sympy import sympify, N, simplify

from app.search_validator import validate_math_question, generate_search_response
from app.lancedb_store import LanceDBStore
from app.embedding_client import get_embedding
from app.ai_gateway import GatewayResponse, AnswerResponse
from app.config import settings

logger = logging.getLogger(__name__)

# Configurable KB threshold via environment variable (default 0.75)
KB_THRESHOLD = float(os.getenv("KB_THRESHOLD", "0.75"))

def evaluate_simple_arithmetic(question: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced arithmetic evaluation for basic math expressions.
    Handles expressions like "What is 2+2?", "Calculate 5*3", etc.
    """
    if not question or not isinstance(question, str):
        return None
    
    # Clean and normalize the question
    q = question.strip().lower()
    
    # Remove common prefixes
    prefixes = ["what is", "calculate", "solve", "find", "compute", "evaluate"]
    for prefix in prefixes:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break
    
    # Remove question marks and other punctuation
    q = re.sub(r'[?!.]+$', '', q).strip()
    
    # Check if it's a simple arithmetic expression
    # Allow digits, basic operators, parentheses, spaces, and decimal points
    if re.match(r'^[\d\s\+\-\*\/\^\(\)\.\,]+$', q):
        try:
            # Replace common patterns
            expr = q.replace('^', '**')  # Handle exponents
            expr = expr.replace(',', '')  # Remove commas from numbers
            expr = re.sub(r'\s+', '', expr)  # Remove spaces
            
            # Use SymPy to safely evaluate
            result = sympify(expr)
            numeric_result = N(result)
            
            # Create step-by-step solution
            steps = [
                f"Given expression: {question}",
                f"Simplified expression: {expr.replace('**', '^')}",
                f"Calculating: {result}",
                f"Final answer: {numeric_result}"
            ]
            
            return {
                "answer": str(numeric_result),
                "steps": steps,
                "source": "arithmetic_calculator",
                "score": 1.0,
                "method": "sympy_evaluation"
            }
            
        except Exception as e:
            logger.debug(f"Arithmetic evaluation failed for '{q}': {e}")
            return None
    
    return None

def detect_mathematical_concepts(question: str) -> Dict[str, Any]:
    """
    Detect mathematical concepts and provide targeted responses.
    """
    q = question.lower()
    
    # Basic arithmetic patterns
    arithmetic_patterns = [
        r'\d+\s*[\+\-\*\/]\s*\d+',
        r'what is.*\d+.*[\+\-\*\/].*\d+',
        r'calculate.*\d+',
        r'solve.*\d+'
    ]
    
    for pattern in arithmetic_patterns:
        if re.search(pattern, q):
            return {"type": "arithmetic", "confidence": 0.9}
    
    # Algebra patterns
    algebra_patterns = [
        r'solve.*[xy].*=',
        r'find.*[xy]',
        r'equation',
        r'variable'
    ]
    
    for pattern in algebra_patterns:
        if re.search(pattern, q):
            return {"type": "algebra", "confidence": 0.8}
    
    # Geometry patterns
    geometry_patterns = [
        r'area.*circle',
        r'volume.*sphere',
        r'triangle',
        r'rectangle',
        r'perimeter'
    ]
    
    for pattern in geometry_patterns:
        if re.search(pattern, q):
            return {"type": "geometry", "confidence": 0.8}
    
    return {"type": "general_math", "confidence": 0.5}

async def solve_with_dspy_cot(question: str) -> Dict[str, Any]:
    """
    Use DSPy Chain of Thought for complex mathematical reasoning.
    """
    try:
        from app.dspy_integration import MathCoTPipeline
        
        pipeline = MathCoTPipeline(api_key=settings.GOOGLE_API_KEY)
        result = pipeline.solve(question)
        
        return {
            "answer": result.get("answer", ""),
            "steps": result.get("steps", []),
            "source": "dspy_chain_of_thought",
            "score": 0.85,
            "method": "llm_reasoning"
        }
    except Exception as e:
        logger.error(f"DSPy CoT failed: {e}")
        return None

async def search_knowledge_base(question: str) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for similar problems.
    """
    try:
        # Initialize LanceDB store
        store = LanceDBStore(
            db_path=os.getenv("LANCEDB_PATH", "lancedb_math"),
            table_name=os.getenv("LANCEDB_TABLE", "math_qa"),
        )
        
        # Check for exact match first
        exact_match = store.find_exact_match(question)
        if exact_match:
            logger.info("Found exact match in knowledge base")
            return [exact_match]
        
        # Semantic search
        embedding = get_embedding(question)
        results = await store.semantic_search(
            query_embedding=embedding,
            top_k=3,
            threshold=KB_THRESHOLD
        )
        
        if results:
            logger.info(f"Found {len(results)} similar problems in knowledge base")
            return results
        
        return []
        
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return []

async def search_web_with_mcp(question: str) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily MCP for mathematical content.
    """
    try:
        from app.mcp_client import web_search_math
        
        results = await web_search_math(question, max_results=3)
        
        web_responses = []
        for result in results:
            web_responses.append({
                "answer": result.snippet,
                "steps": [f"Web search result from: {result.title}", result.snippet],
                "source": "web_search_mcp",
                "score": result.relevance,
                "url": result.url
            })
        
        return web_responses
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []

async def _map_items_to_answer_responses(items: List[Dict[str, Any]]) -> List[AnswerResponse]:
    """Convert internal result format to AnswerResponse objects."""
    results = []
    for item in items:
        results.append(AnswerResponse(
            answer=item.get("answer", ""),
            steps=item.get("steps", []),
            source=item.get("source", "unknown"),
            score=float(item.get("score", 0.0))
        ))
    return results

async def solve_math_problem(question: str) -> GatewayResponse:
    """
    Enhanced routing pipeline with proper arithmetic handling.
    
    Flow:
    1. Input validation and concept detection
    2. Simple arithmetic evaluation (FIRST PRIORITY)
    3. Knowledge base search (exact + semantic)
    4. DSPy Chain of Thought reasoning
    5. Web search via MCP
    6. Fallback to basic web search
    """
    logger.info(f"🔍 Processing question: {question}")
    
    # Step 1: Detect mathematical concepts
    concept_info = detect_mathematical_concepts(question)
    logger.info(f"📊 Detected concept: {concept_info}")
    
    # Step 2: Try simple arithmetic evaluation FIRST
    arithmetic_result = evaluate_simple_arithmetic(question)
    if arithmetic_result:
        logger.info("✅ Solved with arithmetic evaluation")
        return GatewayResponse(
            query=question,
            routed_to="arithmetic_calculator",
            results=[AnswerResponse(
                answer=arithmetic_result["answer"],
                steps=arithmetic_result["steps"],
                source=arithmetic_result["source"],
                score=arithmetic_result["score"]
            )],
            fallback_used=False
        )
    
    # Step 3: Validate if it's a math question
    try:
        is_math = validate_math_question(question)
    except Exception:
        is_math = True  # Assume it's math-related if validation fails
    
    if not is_math:
        logger.warning("❌ Question not recognized as math-related")
        return GatewayResponse(
            query=question,
            routed_to="validation_failed",
            results=[AnswerResponse(
                answer="This doesn't appear to be a mathematical question. Please ask a math-related question.",
                steps=["Validated input as non-mathematical"],
                source="input_validator",
                score=0.0
            )],
            fallback_used=True
        )
    
    # Step 4: Search knowledge base
    kb_results = await search_knowledge_base(question)
    if kb_results:
        logger.info("✅ Found solution in knowledge base")
        results = await _map_items_to_answer_responses(kb_results)
        return GatewayResponse(
            query=question,
            routed_to="knowledge_base",
            results=results,
            fallback_used=False
        )
    
    # Step 5: Try DSPy Chain of Thought
    cot_result = await solve_with_dspy_cot(question)
    if cot_result:
        logger.info("✅ Solved with DSPy Chain of Thought")
        results = await _map_items_to_answer_responses([cot_result])
        return GatewayResponse(
            query=question,
            routed_to="chain_of_thought",
            results=results,
            fallback_used=False
        )
    
    # Step 6: Web search via MCP
    web_results = await search_web_with_mcp(question)
    if web_results:
        logger.info("✅ Found results via web search")
        results = await _map_items_to_answer_responses(web_results)
        return GatewayResponse(
            query=question,
            routed_to="web_search",
            results=results,
            fallback_used=True
        )
    
    # Step 7: Final fallback - basic search response
    logger.warning("⚠️ All methods failed, using basic fallback")
    fallback_items = generate_search_response(question)
    results = await _map_items_to_answer_responses(fallback_items)
    
    return GatewayResponse(
        query=question,
        routed_to="fallback_search",
        results=results,
        fallback_used=True
    )