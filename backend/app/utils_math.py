# backend/app/utils_math.py
from typing import Optional, Tuple
import re
from sympy import sympify, N, binomial
from sympy.core.sympify import SympifyError

# Detect simple arithmetic or combinatorics like "15 choose 3", "C(15,3)", "15C3"
_RE_CHOOSE = re.compile(
    r'^\s*(\d+)\s*(?:choose|[Cc])\s*[\(\s]?\s*(\d+)\s*[\)\s]?\s*$',
    flags=re.IGNORECASE,
)

def _normalize_for_eval(text: str) -> str:
    """
    Normalize by:
      - stripping leading/trailing whitespace
      - removing trailing punctuation like ? ! .
      - removing commas in numbers (e.g. 1,000 -> 1000)
      - collapsing multiple spaces
    """
    t = text.strip()
    # remove question/exclamation marks and trailing periods
    t = t.replace('?', '').replace('!', '').rstrip('.')
    # remove commas in numbers
    t = t.replace(',', '')
    # collapse multiple spaces
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def evaluate_simple_expression(text: str) -> Optional[Tuple[str, float]]:
    """
    Try to evaluate text as a simple arithmetic expression.
    Returns (rendered_answer_string, numeric_value) on success, or None if not recognized.
    Uses SymPy for safe numeric evaluation.
    Recognizes "n choose k" patterns too.
    """
    if not text or not isinstance(text, str):
        return None

    # Normalize first (removes ? ! and commas etc.)
    t = _normalize_for_eval(text)

    # Remove common leading phrases like "what is", "calculate"
    t = re.sub(r'^(what is|calculate|evaluate|find)\s+', '', t, flags=re.IGNORECASE).strip()

    # check for "choose" combos: e.g., "15 choose 3" or "15C3" or "C(15,3)"
    m = _RE_CHOOSE.match(t)
    if m:
        try:
            n = int(m.group(1))
            k = int(m.group(2))
            val = int(binomial(n, k))
            return (f"{val}", float(val))
        except Exception:
            return None

    # For purely numeric/arithmetic expressions only: allow digits/operators, parentheses, decimal, whitespace, percent
    if re.fullmatch(r'[\d\s\.\+\-\*\/\^\(\)%,]+', t):
        try:
            # Convert caret to python power operator
            expr = t.replace('^', '**')
            # Use sympy to evaluate safely
            res = sympify(expr, evaluate=True)
            num = N(res)
            return (str(res), float(num))
        except (SympifyError, ValueError, TypeError, OverflowError):
            return None
    return None
