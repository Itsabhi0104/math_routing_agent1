# backend/app/dspy_integration.py

import logging
import sympy as sp
from typing import Any, Dict
import dspy
from dspy import Signature, ChainOfThought

logger = logging.getLogger(__name__)


class MathProblemSolver(Signature):
    """
    DSPy signature defining inputs and outputs for
    step-by-step mathematical problem solving.
    """
    question: str = dspy.InputField(desc="Mathematical question to solve")
    step_by_step_solution: str = dspy.OutputField(
        desc="Detailed step-by-step solution"
    )
    final_answer: str = dspy.OutputField(
        desc="Final numerical or symbolic answer"
    )


class MathCoTPipeline:
    """
    Configures DSPy with Gemini 1.5 Flash and wraps
    a Chain-of-Thought reasoning pipeline.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str = None,
    ):
        # If api_key is None, DSPy will look for an environment variable
        dspy.configure(model=model_name, api_key=api_key)
        self.cot = ChainOfThought(MathProblemSolver)
        logger.info("Initialized MathCoTPipeline with model %s", model_name)

    def solve(self, question: str) -> Dict[str, Any]:
        """
        Run Chain-of-Thought reasoning on the given math question.
        Returns:
            {
                "steps": List[str],       # ordered reasoning steps
                "answer": str             # final answer expression
            }
        """
        try:
            # Execute the signature; returns a dict matching the Signature fields
            result = self.cot.run(question=question)
            # Split the multi-line solution into individual steps
            steps = result["step_by_step_solution"].splitlines()
            answer = result["final_answer"]

            # Verify the final answer with SymPy
            if not self._verify(question, answer):
                logger.warning(
                    "SymPy verification failed for answer: %s", answer
                )
                raise ValueError("Answer verification failed")

            return {"steps": steps, "answer": answer}

        except Exception:
            logger.exception("MathCoTPipeline failed for question: %s", question)
            raise

    def _verify(self, question: str, answer: str) -> bool:
        """
        Use SymPy to parse and verify the final answer.
        - If the question contains an equation ("="), ensure LHS == RHS.
        - Otherwise, simply ensure `answer` is a valid SymPy expression.
        """
        try:
            expr = sp.sympify(answer)
            if "=" in question:
                lhs_str, rhs_str = question.split("=", 1)
                lhs = sp.sympify(lhs_str.strip())
                # Check that lhs - answer == 0
                return sp.simplify(lhs - expr) == 0
            return True
        except Exception:
            logger.debug("SymPy failed to parse/verify: %s", answer)
            return False


# Manual smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = MathCoTPipeline(api_key=None)
    sample_q = "Solve 2*x + 3 = 7 for x"
    output = pipeline.solve(sample_q)
    print("Steps:")
    for idx, step in enumerate(output["steps"], start=1):
        print(f"  {idx}. {step}")
    print("Answer:", output["answer"])
