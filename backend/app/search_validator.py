# backend/app/search_validator.py

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp

logger = logging.getLogger(__name__)

class MathQuestionValidator:
    """
    Enhanced validator for mathematical questions with comprehensive pattern recognition.
    """
    
    # Mathematical patterns and indicators
    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/\^\=]\s*\d+',  # Basic arithmetic: 2+2, 5*3
        r'\b(solve|find|calculate|compute|evaluate|determine)\b',  # Action verbs
        r'\b(equation|formula|expression|function)\b',  # Math terms
        r'\b(derivative|integral|limit|sum|product)\b',  # Calculus terms
        r'\b(triangle|circle|square|rectangle|polygon|sphere|cube)\b',  # Geometry
        r'\b(sine|cosine|tangent|logarithm|exponential)\b',  # Trigonometry
        r'\b(matrix|vector|determinant|eigenvalue)\b',  # Linear algebra
        r'\b(probability|statistics|mean|median|variance)\b',  # Statistics
        r'[xy]\s*[\+\-\*\/\=]',  # Variable expressions: x+2, y=3
        r'\b\d+\s*(choose|C)\s*\d+\b',  # Combinatorics: 5 choose 2
        r'∫|∑|π|√|∞|≤|≥|≠|±|°',  # Mathematical symbols
        r'\b(what\s+is|how\s+much|find\s+the)\b.*\d',  # Question patterns with numbers
    ]
    
    # Keywords that strongly indicate mathematical content
    STRONG_MATH_INDICATORS = [
        'mathematics', 'math', 'algebra', 'geometry', 'calculus', 'trigonometry',
        'arithmetic', 'equation', 'formula', 'theorem', 'proof', 'solution'
    ]
    
    # Educational math question patterns
    EDUCATIONAL_PATTERNS = [
        r'\bstep\s+by\s+step\b',
        r'\bhow\s+to\s+(solve|calculate|find)\b',
        r'\bexplain\s+(how|why|the)\b',
        r'\bshow\s+that\b',
        r'\bprove\s+that\b'
    ]
    
    # Non-mathematical content patterns to exclude
    EXCLUSION_PATTERNS = [
        r'\b(politics|political|election|vote|government)\b',
        r'\b(religion|religious|god|allah|jesus|buddha)\b',
        r'\b(sex|sexual|porn|nude|naked)\b',
        r'\b(drugs|marijuana|cocaine|alcohol)\b',
        r'\b(violence|kill|murder|harm|death)\b',
        r'\b(hack|exploit|virus|malware|password)\b'
    ]

    @classmethod
    def validate_math_question(cls, text: str) -> bool:
        """
        Validate if a question is mathematical and appropriate.
        
        Args:
            text: Input text to validate
            
        Returns:
            True if the text appears to be a valid mathematical question
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 2:
            return False
        
        text_clean = text.strip().lower()
        
        # Check for excluded content first
        for pattern in cls.EXCLUSION_PATTERNS:
            if re.search(pattern, text_clean, re.IGNORECASE):
                logger.debug(f"Excluded content detected: {pattern}")
                return False
        
        # Calculate mathematical content score
        math_score = cls._calculate_math_score(text_clean)
        
        # Determine if it's mathematical based on score
        is_mathematical = math_score >= 1.0
        
        logger.debug(f"Math validation: '{text[:50]}...' -> score={math_score:.2f}, valid={is_mathematical}")
        
        return is_mathematical
    
    @classmethod
    def _calculate_math_score(cls, text: str) -> float:
        """
        Calculate a numerical score indicating how mathematical the text is.
        
        Args:
            text: Cleaned text to analyze
            
        Returns:
            Mathematical score (higher = more mathematical)
        """
        score = 0.0
        
        # Check strong mathematical indicators (high weight)
        for indicator in cls.STRONG_MATH_INDICATORS:
            if indicator in text:
                score += 2.0
                break  # Don't double-count
        
        # Check mathematical patterns (medium weight)
        pattern_matches = 0
        for pattern in cls.MATH_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches += 1
        
        score += min(pattern_matches * 0.5, 2.0)  # Cap at 2.0
        
        # Check for educational indicators (low weight)
        for pattern in cls.EDUCATIONAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3
        
        # Basic number presence (very low weight)
        if re.search(r'\d', text):
            score += 0.2
        
        # Question word bonus
        question_words = ['what', 'how', 'find', 'calculate', 'solve', 'determine']
        if any(word in text for word in question_words):
            score += 0.3
        
        return score
    
    @classmethod
    def get_question_type(cls, text: str) -> str:
        """
        Determine the type of mathematical question.
        
        Args:
            text: Mathematical question text
            
        Returns:
            Question type category
        """
        text_lower = text.lower()
        
        # Arithmetic
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', text) or 'calculate' in text_lower:
            return 'arithmetic'
        
        # Algebra
        if re.search(r'[xy]\s*[\+\-\*\/\=]', text) or 'solve' in text_lower and 'equation' in text_lower:
            return 'algebra'
        
        # Geometry
        geometry_terms = ['triangle', 'circle', 'area', 'volume', 'perimeter', 'angle']
        if any(term in text_lower for term in geometry_terms):
            return 'geometry'
        
        # Calculus
        calculus_terms = ['derivative', 'integral', 'limit', 'differential']
        if any(term in text_lower for term in calculus_terms):
            return 'calculus'
        
        # Trigonometry
        trig_terms = ['sine', 'cosine', 'tangent', 'sin', 'cos', 'tan']
        if any(term in text_lower for term in trig_terms):
            return 'trigonometry'
        
        # Statistics
        stats_terms = ['probability', 'mean', 'median', 'variance', 'standard deviation']
        if any(term in text_lower for term in stats_terms):
            return 'statistics'
        
        return 'general_math'
    
    @classmethod
    def extract_mathematical_elements(cls, text: str) -> Dict[str, List[str]]:
        """
        Extract mathematical elements from the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing extracted mathematical elements
        """
        elements = {
            'numbers': [],
            'variables': [],
            'operators': [],
            'functions': [],
            'expressions': []
        }
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        elements['numbers'] = list(set(numbers))
        
        # Extract variables (single letters)
        variables = re.findall(r'\b[a-zA-Z]\b', text)
        elements['variables'] = list(set(variables))
        
        # Extract operators
        operators = re.findall(r'[\+\-\*\/\=\^\<\>]', text)
        elements['operators'] = list(set(operators))
        
        # Extract function names
        function_pattern = r'\b(sin|cos|tan|log|ln|exp|sqrt|abs)\b'
        functions = re.findall(function_pattern, text, re.IGNORECASE)
        elements['functions'] = list(set(functions))
        
        # Extract mathematical expressions
        expr_pattern = r'\b\w*[a-zA-Z]\w*\s*[\+\-\*\/\=\^]\s*\w+\b'
        expressions = re.findall(expr_pattern, text)
        elements['expressions'] = list(set(expressions))
        
        return elements

# Backward compatibility function
def validate_math_question(text: str) -> bool:
    """Validate if text is a mathematical question."""
    return MathQuestionValidator.validate_math_question(text)

class SearchResponseGenerator:
    """
    Generate appropriate search responses for mathematical queries.
    """
    
    @staticmethod
    def generate_search_response(query: str) -> List[Dict[str, Any]]:
        """
        Generate a structured search response for mathematical queries.
        
        Args:
            query: Mathematical query
            
        Returns:
            List of search result dictionaries
        """
        # Determine question type
        question_type = MathQuestionValidator.get_question_type(query)
        
        # Generate contextual response based on type
        if question_type == 'arithmetic':
            return SearchResponseGenerator._generate_arithmetic_response(query)
        elif question_type == 'algebra':
            return SearchResponseGenerator._generate_algebra_response(query)
        elif question_type == 'geometry':
            return SearchResponseGenerator._generate_geometry_response(query)
        elif question_type == 'calculus':
            return SearchResponseGenerator._generate_calculus_response(query)
        else:
            return SearchResponseGenerator._generate_general_response(query)
    
    @staticmethod
    def _generate_arithmetic_response(query: str) -> List[Dict[str, Any]]:
        """Generate response for arithmetic questions."""
        return [{
            "answer": f"Arithmetic calculation for: {query}",
            "steps": [
                "Identify the arithmetic operation",
                "Apply the appropriate calculation method",
                "Verify the result"
            ],
            "score": 0.7,
            "source": "arithmetic_search",
            "confidence": "medium"
        }]
    
    @staticmethod 
    def _generate_algebra_response(query: str) -> List[Dict[str, Any]]:
        """Generate response for algebra questions."""
        return [{
            "answer": f"Algebraic solution for: {query}",
            "steps": [
                "Set up the equation",
                "Isolate the variable",
                "Solve for the unknown",
                "Check the solution"
            ],
            "score": 0.6,
            "source": "algebra_search",
            "confidence": "medium"
        }]
    
    @staticmethod
    def _generate_geometry_response(query: str) -> List[Dict[str, Any]]:
        """Generate response for geometry questions."""
        return [{
            "answer": f"Geometric solution for: {query}",
            "steps": [
                "Identify the geometric shape or property",
                "Apply the relevant formula",
                "Calculate the result",
                "Include appropriate units"
            ],
            "score": 0.6,
            "source": "geometry_search",
            "confidence": "medium"
        }]
    
    @staticmethod
    def _generate_calculus_response(query: str) -> List[Dict[str, Any]]:
        """Generate response for calculus questions."""
        return [{
            "answer": f"Calculus solution for: {query}",
            "steps": [
                "Identify the calculus operation (derivative/integral)",
                "Apply the appropriate rules",
                "Simplify the expression",
                "State the final result"
            ],
            "score": 0.5,
            "source": "calculus_search",
            "confidence": "low"
        }]
    
    @staticmethod
    def _generate_general_response(query: str) -> List[Dict[str, Any]]:
        """Generate response for general mathematical questions."""
        return [{
            "answer": f"Mathematical solution for: {query}",
            "steps": [
                "Analyze the mathematical problem",
                "Determine the appropriate method",
                "Apply mathematical principles",
                "Verify the solution"
            ],
            "score": 0.4,
            "source": "general_math_search",
            "confidence": "low"
        }]

# Backward compatibility function
def generate_search_response(query: str) -> List[Dict[str, Any]]:
    """Generate search response for a mathematical query."""
    return SearchResponseGenerator.generate_search_response(query)

# Utility functions
def is_equation(text: str) -> bool:
    """Check if text contains a mathematical equation."""
    return '=' in text and re.search(r'\w+\s*=\s*\w+', text) is not None

def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    try:
        number_strings = re.findall(r'-?\d+(?:\.\d+)?', text)
        return [float(num) for num in number_strings]
    except ValueError:
        return []

def contains_variables(text: str) -> bool:
    """Check if text contains mathematical variables."""
    return re.search(r'\b[a-zA-Z]\b', text) is not None

# Example usage and testing
if __name__ == "__main__":
    test_questions = [
        "What is 2+2?",
        "Solve x + 5 = 10 for x",
        "Find the area of a circle with radius 3",
        "What is the derivative of x^2?",
        "How do I cook pasta?",  # Non-mathematical
        "Tell me about politics"  # Excluded content
    ]
    
    validator = MathQuestionValidator()
    
    for question in test_questions:
        is_valid = validator.validate_math_question(question)
        question_type = validator.get_question_type(question)
        print(f"'{question}' -> Valid: {is_valid}, Type: {question_type}")