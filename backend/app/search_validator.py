# backend/app/search_validator.py

import re
from typing import Any, Dict, List

def validate_math_question(text: str) -> bool:
    """
    Basic heuristic to check if a question is math-related.
    Returns True if it contains digits or common math operators/symbols.
    """
    math_pattern = re.compile(r'[\d\+\-\*\/\=\^\(\)]')
    return bool(math_pattern.search(text))


def generate_search_response(query: str) -> List[Dict[str, Any]]:
    """
    Stubbed web-search fallback. MUST return a list of dicts where each dict
    has the keys expected by AnswerResponse: 'answer', 'steps', 'score', 'source'.
    Replace with your real web-search integration when ready.
    """
    # Example single-result fallback (expand to many results if your search returns more)
    return [
        {
            "answer": f"Search result for: {query}",
            "steps": ["Performed web search", "Retrieved search results"],
            "score": 0.5,
            "source": "web_search",
        }
    ]
