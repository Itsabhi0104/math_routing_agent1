#!/usr/bin/env python3
"""
JEE Benchmark Evaluation System
Evaluates the Math Routing Agent against JEE (Joint Entrance Examination) mathematics problems.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JEEProblem:
    """Represents a JEE mathematics problem."""
    id: str
    question: str
    correct_answer: str
    topic: str
    difficulty: str  # easy, medium, hard
    year: Optional[int] = None
    source: str = "JEE"

@dataclass
class BenchmarkResult:
    """Results from evaluating a single problem."""
    problem_id: str
    question: str
    expected_answer: str
    predicted_answer: str
    is_correct: bool
    confidence_score: float
    response_time: float
    route_taken: str
    steps_count: int
    topic: str
    difficulty: str

class JEEDatasetLoader:
    """Loads JEE mathematics problems for benchmarking."""
    
    def __init__(self):
        self.problems: List[JEEProblem] = []
    
    def load_sample_problems(self) -> List[JEEProblem]:
        """
        Load sample JEE problems covering different mathematical topics.
        In production, this would load from a comprehensive JEE dataset.
        """
        sample_problems = [
            # Arithmetic & Basic Algebra
            JEEProblem(
                id="jee_001",
                question="If x + y = 10 and x - y = 2, find the value of x.",
                correct_answer="6",
                topic="algebra",
                difficulty="easy"
            ),
            JEEProblem(
                id="jee_002", 
                question="Find the value of 2^5 × 3^2",
                correct_answer="288",
                topic="arithmetic",
                difficulty="easy"
            ),
            
            # Quadratic Equations
            JEEProblem(
                id="jee_003",
                question="Solve the quadratic equation x² - 5x + 6 = 0",
                correct_answer="x = 2, 3",
                topic="algebra",
                difficulty="medium"
            ),
            
            # Geometry
            JEEProblem(
                id="jee_004",
                question="Find the area of a circle with diameter 14 cm. Use π = 22/7",
                correct_answer="154",
                topic="geometry",
                difficulty="easy"
            ),
            JEEProblem(
                id="jee_005",
                question="In a right triangle, if one angle is 30°, and the hypotenuse is 10, find the length of the side opposite to the 30° angle.",
                correct_answer="5",
                topic="trigonometry",
                difficulty="medium"
            ),
            
            # Calculus
            JEEProblem(
                id="jee_006",
                question="Find the derivative of f(x) = x³ + 2x² - 5x + 1",
                correct_answer="3x² + 4x - 5",
                topic="calculus",
                difficulty="medium"
            ),
            JEEProblem(
                id="jee_007",
                question="Evaluate the integral ∫(2x + 3)dx",
                correct_answer="x² + 3x + C",
                topic="calculus",
                difficulty="medium"
            ),
            
            # Complex Problems
            JEEProblem(
                id="jee_008",
                question="If the roots of the equation x² + px + q = 0 are α and β, find the value of α² + β² in terms of p and q.",
                correct_answer="p² - 2q",
                topic="algebra",
                difficulty="hard"
            ),
            JEEProblem(
                id="jee_009",
                question="Find the number of ways to arrange 5 different books on a shelf such that 2 specific books are always together.",
                correct_answer="48",
                topic="combinatorics",
                difficulty="medium"
            ),
            
            # Advanced Topics
            JEEProblem(
                id="jee_010",
                question="If z₁ = 3 + 4i and z₂ = 1 - 2i, find |z₁ + z₂|",
                correct_answer="√20",
                topic="complex_numbers",
                difficulty="hard"
            )
        ]
        
        self.problems = sample_problems
        logger.info(f"Loaded {len(sample_problems)} JEE problems")
        return sample_problems

class AnswerValidator:
    """Validates predicted answers against expected answers."""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        
        # Remove extra whitespace and convert to lowercase
        normalized = answer.strip().lower()
        
        # Handle common mathematical notations
        normalized = normalized.replace("×", "*")
        normalized = normalized.replace("÷", "/")
        normalized = normalized.replace("π", "pi")
        
        return normalized
    
    @staticmethod
    def extract_numeric_value(answer: str) -> Optional[float]:
        """Extract numeric value from answer string."""
        import re
        
        # Look for numbers (including decimals)
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            try:
                return float(numbers[-1])  # Return the last number found
            except ValueError:
                return None
        return None
    
    def is_correct(self, predicted: str, expected: str) -> bool:
        """
        Determine if predicted answer matches expected answer.
        Uses multiple validation strategies.
        """
        pred_norm = self.normalize_answer(predicted)
        exp_norm = self.normalize_answer(expected)
        
        # Strategy 1: Exact string match
        if pred_norm == exp_norm:
            return True
        
        # Strategy 2: Numeric comparison
        pred_num = self.extract_numeric_value(predicted)
        exp_num = self.extract_numeric_value(expected)
        
        if pred_num is not None and exp_num is not None:
            # Allow small floating point differences
            return abs(pred_num - exp_num) < 1e-6
        
        # Strategy 3: Contains expected answer
        if exp_norm in pred_norm or pred_norm in exp_norm:
            return True
        
        # Strategy 4: Multiple choice handling (for answers like "x = 2, 3")
        if "," in expected:
            exp_parts = [part.strip() for part in expected.split(",")]
            return any(self.normalize_answer(part) in pred_norm for part in exp_parts)
        
        return False

class JEEBenchmarkEvaluator:
    """Main benchmark evaluation system."""
    
    def __init__(self):
        self.dataset_loader = JEEDatasetLoader()
        self.validator = AnswerValidator()
        self.results: List[BenchmarkResult] = []
    
    async def evaluate_single_problem(self, problem: JEEProblem) -> BenchmarkResult:
        """Evaluate the agent on a single JEE problem."""
        logger.info(f"Evaluating problem {problem.id}: {problem.question[:50]}...")
        
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from app.orchestration import solve_math_problem
            
            # Get agent's response
            response = await solve_math_problem(problem.question)
            
            response_time = time.time() - start_time
            
            # Extract answer and metadata
            if response.results:
                primary_result = response.results[0]
                predicted_answer = primary_result.answer
                confidence_score = primary_result.score
                route_taken = response.routed_to
                steps_count = len(primary_result.steps) if primary_result.steps else 0
            else:
                predicted_answer = ""
                confidence_score = 0.0
                route_taken = "no_response"
                steps_count = 0
            
            # Validate correctness
            is_correct = self.validator.is_correct(predicted_answer, problem.correct_answer)
            
            result = BenchmarkResult(
                problem_id=problem.id,
                question=problem.question,
                expected_answer=problem.correct_answer,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                confidence_score=confidence_score,
                response_time=response_time,
                route_taken=route_taken,
                steps_count=steps_count,
                topic=problem.topic,
                difficulty=problem.difficulty
            )
            
            logger.info(f"Problem {problem.id}: {'✅ CORRECT' if is_correct else '❌ INCORRECT'} "
                       f"(Expected: {problem.correct_answer}, Got: {predicted_answer})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating problem {problem.id}: {e}")
            
            return BenchmarkResult(
                problem_id=problem.id,
                question=problem.question,
                expected_answer=problem.correct_answer,
                predicted_answer=f"ERROR: {str(e)}",
                is_correct=False,
                confidence_score=0.0,
                response_time=time.time() - start_time,
                route_taken="error",
                steps_count=0,
                topic=problem.topic,
                difficulty=problem.difficulty
            )
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark evaluation."""
        logger.info("🚀 Starting JEE Benchmark Evaluation")
        
        # Load problems
        problems = self.dataset_loader.load_sample_problems()
        
        # Evaluate each problem
        self.results = []
        for problem in problems:
            result = await self.evaluate_single_problem(problem)
            self.results.append(result)
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.5)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        logger.info("✅ JEE Benchmark Evaluation Complete")
        return metrics
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive benchmark metrics."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        total_problems = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        
        # Overall accuracy
        overall_accuracy = correct_answers / total_problems if total_problems > 0 else 0
        
        # Accuracy by topic
        topic_stats = {}
        for result in self.results:
            topic = result.topic
            if topic not in topic_stats:
                topic_stats[topic] = {"total": 0, "correct": 0}
            topic_stats[topic]["total"] += 1
            if result.is_correct:
                topic_stats[topic]["correct"] += 1
        
        for topic in topic_stats:
            topic_stats[topic]["accuracy"] = (
                topic_stats[topic]["correct"] / topic_stats[topic]["total"]
                if topic_stats[topic]["total"] > 0 else 0
            )
        
        # Accuracy by difficulty
        difficulty_stats = {}
        for result in self.results:
            difficulty = result.difficulty
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"total": 0, "correct": 0}
            difficulty_stats[difficulty]["total"] += 1
            if result.is_correct:
                difficulty_stats[difficulty]["correct"] += 1
        
        for difficulty in difficulty_stats:
            difficulty_stats[difficulty]["accuracy"] = (
                difficulty_stats[difficulty]["correct"] / difficulty_stats[difficulty]["total"]
                if difficulty_stats[difficulty]["total"] > 0 else 0
            )
        
        # Route analysis
        route_stats = {}
        for result in self.results:
            route = result.route_taken
            if route not in route_stats:
                route_stats[route] = {"total": 0, "correct": 0}
            route_stats[route]["total"] += 1
            if result.is_correct:
                route_stats[route]["correct"] += 1
        
        for route in route_stats:
            route_stats[route]["accuracy"] = (
                route_stats[route]["correct"] / route_stats[route]["total"]
                if route_stats[route]["total"] > 0 else 0
            )
        
        # Performance metrics
        response_times = [r.response_time for r in self.results]
        avg_response_time = np.mean(response_times)
        median_response_time = np.median(response_times)
        
        avg_confidence = np.mean([r.confidence_score for r in self.results])
        avg_steps = np.mean([r.steps_count for r in self.results])
        
        # Detailed results
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                "problem_id": result.problem_id,
                "question": result.question[:100] + "..." if len(result.question) > 100 else result.question,
                "expected": result.expected_answer,
                "predicted": result.predicted_answer,
                "correct": result.is_correct,
                "confidence": result.confidence_score,
                "response_time": result.response_time,
                "route": result.route_taken,
                "topic": result.topic,
                "difficulty": result.difficulty
            })
        
        return {
            "benchmark_summary": {
                "total_problems": total_problems,
                "correct_answers": correct_answers,
                "overall_accuracy": overall_accuracy,
                "avg_response_time": avg_response_time,
                "median_response_time": median_response_time,
                "avg_confidence_score": avg_confidence,
                "avg_steps_per_solution": avg_steps
            },
            "topic_performance": topic_stats,
            "difficulty_performance": difficulty_stats,
            "route_performance": route_stats,
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_report(self, metrics: Dict[str, Any], output_path: str = "jee_benchmark_report.json"):
        """Generate a comprehensive benchmark report."""
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Benchmark report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save benchmark report: {e}")
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print a formatted summary of benchmark results."""
        summary = metrics["benchmark_summary"]
        
        print("\n" + "="*60)
        print("           JEE BENCHMARK EVALUATION RESULTS")
        print("="*60)
        print(f"Total Problems Evaluated: {summary['total_problems']}")
        print(f"Correct Answers: {summary['correct_answers']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
        print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
        print(f"Average Confidence Score: {summary['avg_confidence_score']:.2f}")
        print(f"Average Steps per Solution: {summary['avg_steps_per_solution']:.1f}")
        
        print("\n📊 Performance by Topic:")
        for topic, stats in metrics["topic_performance"].items():
            print(f"  {topic.title()}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        print("\n📈 Performance by Difficulty:")
        for difficulty, stats in metrics["difficulty_performance"].items():
            print(f"  {difficulty.title()}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        print("\n🛤️  Performance by Route:")
        for route, stats in metrics["route_performance"].items():
            print(f"  {route}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        print("\n💡 Top Performing Areas:")
        topic_accuracies = [(topic, stats['accuracy']) for topic, stats in metrics["topic_performance"].items()]
        top_topics = sorted(topic_accuracies, key=lambda x: x[1], reverse=True)[:3]
        for topic, accuracy in top_topics:
            print(f"  ✅ {topic.title()}: {accuracy:.2%}")
        
        print("\n⚠️  Areas for Improvement:")
        bottom_topics = sorted(topic_accuracies, key=lambda x: x[1])[:3]
        for topic, accuracy in bottom_topics:
            if accuracy < 0.8:  # Only show if accuracy is below 80%
                print(f"  🔄 {topic.title()}: {accuracy:.2%}")
        
        print("="*60)

# Main execution function
async def run_jee_benchmark():
    """Main function to run the JEE benchmark evaluation."""
    evaluator = JEEBenchmarkEvaluator()
    
    try:
        # Run the benchmark
        metrics = await evaluator.run_full_benchmark()
        
        # Print summary
        evaluator.print_summary(metrics)
        
        # Generate report
        evaluator.generate_report(metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Benchmark evaluation failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    import os
    
    # Add the backend directory to Python path
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, backend_dir)
    
    # Run the benchmark
    asyncio.run(run_jee_benchmark())