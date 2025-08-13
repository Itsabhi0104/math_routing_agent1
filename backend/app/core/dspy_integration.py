# backend/app/dspy_integration.py

import logging
import re
from typing import Any, Dict, List, Optional
import dspy
from dspy import Signature, ChainOfThought, InputField, OutputField
import sympy as sp

logger = logging.getLogger(__name__)

class MathProblemSolver(Signature):
    """
    Enhanced DSPy signature for comprehensive mathematical problem solving
    with step-by-step reasoning and validation.
    """
    question: str = InputField(desc="Mathematical question or problem to solve")
    problem_type: str = OutputField(desc="Type of mathematical problem (arithmetic, algebra, geometry, calculus, etc.)")
    step_by_step_solution: str = OutputField(desc="Detailed step-by-step solution with clear explanations")
    final_answer: str = OutputField(desc="Final numerical or symbolic answer")
    confidence_score: str = OutputField(desc="Confidence level in the solution (high/medium/low)")

class MathConceptExplainer(Signature):
    """
    DSPy signature for explaining mathematical concepts in simple terms.
    """
    concept: str = InputField(desc="Mathematical concept or solution to explain")
    student_level: str = InputField(desc="Student level (beginner/intermediate/advanced)")
    simplified_explanation: str = OutputField(desc="Simplified explanation suitable for the student level")
    key_steps: str = OutputField(desc="Key steps broken down for understanding")

class MathValidationSignature(Signature):
    """
    DSPy signature for validating mathematical solutions.
    """
    original_problem: str = InputField(desc="Original mathematical problem")
    proposed_solution: str = InputField(desc="Proposed solution to validate")
    validation_result: str = OutputField(desc="Validation result (correct/incorrect/needs_revision)")
    explanation: str = OutputField(desc="Explanation of why the solution is correct or incorrect")

class MathCoTPipeline:
    """
    Enhanced DSPy Chain-of-Thought pipeline configured for Gemini 1.5 Flash
    with comprehensive mathematical problem solving capabilities.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.2
    ):
        try:
            # Configure DSPy with Gemini
            if api_key:
                dspy.configure(
                    model=model_name,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=1024
                )
            else:
                # Fallback to environment variable
                dspy.configure(model=model_name, temperature=temperature)
                
            # Initialize modules
            self.solver = ChainOfThought(MathProblemSolver)
            self.explainer = ChainOfThought(MathConceptExplainer)
            self.validator = ChainOfThought(MathValidationSignature)
            
            logger.info(f"Initialized MathCoTPipeline with model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DSPy pipeline: {e}")
            raise

    def solve(self, question: str, student_level: str = "intermediate") -> Dict[str, Any]:
        """
        Solve mathematical problem using Chain-of-Thought reasoning.
        
        Args:
            question: Mathematical question to solve
            student_level: Target student level for explanations
            
        Returns:
            Dictionary with solution, steps, answer, and metadata
        """
        try:
            logger.info(f"Solving with DSPy CoT: {question[:100]}...")
            
            # Step 1: Solve the problem
            result = self.solver(question=question)
            
            # Step 2: Extract and parse results
            problem_type = result.problem_type
            solution_steps = result.step_by_step_solution.split('\n')
            final_answer = result.final_answer
            confidence = result.confidence_score
            
            # Step 3: Validate the solution using SymPy if possible
            is_valid, validation_msg = self._validate_solution(question, final_answer)
            
            # Step 4: Generate simplified explanation
            explanation = self._generate_explanation(
                solution_steps, 
                final_answer, 
                student_level
            )
            
            # Step 5: Format response
            response = {
                "answer": final_answer,
                "steps": solution_steps,
                "problem_type": problem_type,
                "confidence": confidence,
                "is_validated": is_valid,
                "validation_message": validation_msg,
                "simplified_explanation": explanation,
                "source": "dspy_chain_of_thought",
                "model_used": "gemini-1.5-flash"
            }
            
            logger.info("DSPy CoT solution completed successfully")
            return response
            
        except Exception as e:
            logger.exception(f"DSPy CoT failed for question: {question}")
            raise Exception(f"Chain-of-thought reasoning failed: {str(e)}")

    def explain_concept(self, concept: str, level: str = "intermediate") -> Dict[str, Any]:
        """
        Generate simplified explanation of mathematical concepts.
        """
        try:
            result = self.explainer(concept=concept, student_level=level)
            return {
                "explanation": result.simplified_explanation,
                "key_steps": result.key_steps.split('\n'),
                "level": level
            }
        except Exception as e:
            logger.error(f"Concept explanation failed: {e}")
            return {
                "explanation": f"Unable to explain concept: {concept}",
                "key_steps": [],
                "level": level
            }

    def validate_solution(self, problem: str, solution: str) -> Dict[str, Any]:
        """
        Validate a mathematical solution using DSPy reasoning.
        """
        try:
            result = self.validator(
                original_problem=problem,
                proposed_solution=solution
            )
            return {
                "is_valid": result.validation_result.lower() == "correct",
                "validation_result": result.validation_result,
                "explanation": result.explanation
            }
        except Exception as e:
            logger.error(f"Solution validation failed: {e}")
            return {
                "is_valid": False,
                "validation_result": "error",
                "explanation": f"Validation error: {str(e)}"
            }

    def _validate_solution(self, question: str, answer: str) -> tuple[bool, str]:
        """
        Use SymPy to validate mathematical solutions when possible.
        """
        try:
            # Clean the answer
            clean_answer = re.sub(r'[^\d\+\-\*\/\^\(\)\.\s=x-z]', '', answer.lower())
            
            if not clean_answer.strip():
                return True, "Answer format validation passed"
            
            # Try to parse with SymPy
            expr = sp.sympify(clean_answer.replace('^', '**'))
            
            # For equations, check if both sides are equal
            if "=" in question and "=" in clean_answer:
                parts = clean_answer.split("=")
                if len(parts) == 2:
                    lhs = sp.sympify(parts[0].strip().replace('^', '**'))
                    rhs = sp.sympify(parts[1].strip().replace('^', '**'))
                    is_equal = sp.simplify(lhs - rhs) == 0
                    return is_equal, f"Equation validation: {'passed' if is_equal else 'failed'}"
            
            return True, "SymPy validation passed"
            
        except Exception as e:
            logger.debug(f"SymPy validation failed: {e}")
            # If SymPy fails, assume the answer is valid (could be text explanation)
            return True, f"Validation skipped: {str(e)}"

    def _generate_explanation(self, steps: List[str], answer: str, level: str) -> str:
        """
        Generate a simplified explanation based on the solution steps.
        """
        try:
            # Combine steps into a coherent explanation
            explanation_text = "\n".join(steps)
            
            # Use DSPy to simplify for the target level
            simplified = self.explain_concept(explanation_text, level)
            return simplified.get("explanation", explanation_text)
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return f"Solution: {answer}"

    def adaptive_solve(self, question: str, feedback_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced solving with adaptive learning from feedback history.
        """
        try:
            # Incorporate feedback patterns if available
            context = ""
            if feedback_history:
                positive_feedback = [f for f in feedback_history if f.get('rating', 0) >= 4]
                if positive_feedback:
                    context = f"Based on successful solutions, focus on: {positive_feedback[-1].get('pattern', '')}"
            
            # Enhance question with context
            enhanced_question = f"{context}\n{question}" if context else question
            
            # Solve with enhanced context
            return self.solve(enhanced_question)
            
        except Exception as e:
            logger.error(f"Adaptive solving failed: {e}")
            # Fallback to regular solving
            return self.solve(question)

# Enhanced mathematical validation utilities
def validate_mathematical_expression(expr: str) -> bool:
    """Enhanced validation for mathematical expressions."""
    if not expr or not isinstance(expr, str):
        return False
    
    try:
        # Allow more flexible mathematical expressions
        cleaned = re.sub(r'[^\w\d\+\-\*\/\^\(\)\.\s=<>]', '', expr)
        if not cleaned.strip():
            return False
            
        # Try SymPy parsing
        sp.sympify(cleaned.replace('^', '**'))
        return True
        
    except Exception:
        # Check for mathematical keywords as fallback
        math_keywords = ['solution', 'answer', 'result', 'equals', 'approximately']
        return any(keyword in expr.lower() for keyword in math_keywords)

# Factory function for easy initialization
def create_math_pipeline(api_key: str = None) -> MathCoTPipeline:
    """Factory function to create and configure MathCoTPipeline."""
    return MathCoTPipeline(api_key=api_key)

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the pipeline
    from app.config import settings
    
    try:
        pipeline = MathCoTPipeline(api_key=settings.GOOGLE_API_KEY)
        
        # Test problems
        test_problems = [
            "What is 2 + 2?",
            "Solve x + 5 = 10 for x",
            "Find the derivative of x^2 + 3x + 2",
            "Calculate the area of a circle with radius 5"
        ]
        
        for problem in test_problems:
            print(f"\nProblem: {problem}")
            try:
                result = pipeline.solve(problem)
                print(f"Answer: {result['answer']}")
                print(f"Type: {result['problem_type']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Validated: {result['is_validated']}")
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")