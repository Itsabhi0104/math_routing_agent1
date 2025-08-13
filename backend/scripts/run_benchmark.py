#!/usr/bin/env python3
"""
JEE Benchmark Runner Script
Executes the complete JEE benchmark evaluation for the Math Routing Agent.
"""

import asyncio
import logging
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.benchmarks.jee_benchmark import run_jee_benchmark
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are met for benchmark execution."""
    logger.info("🔍 Checking prerequisites...")
    
    # Check API keys
    if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "your-google-api-key":
        logger.error("❌ GOOGLE_API_KEY not properly configured")
        return False
    
    # Check if knowledge base exists
    if not os.path.exists(settings.LANCEDB_PATH):
        logger.warning(f"⚠️ Knowledge base not found at {settings.LANCEDB_PATH}")
        logger.info("Consider running: python scripts/setup_kb.py")
    
    # Check if the main application is accessible
    try:
        from app.orchestration import solve_math_problem
        logger.info("✅ Math orchestration module accessible")
    except ImportError as e:
        logger.error(f"❌ Cannot import orchestration module: {e}")
        return False
    
    logger.info("✅ Prerequisites check passed")
    return True

async def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="Run JEE Benchmark Evaluation")
    parser.add_argument(
        "--output", 
        default="jee_benchmark_report.json",
        help="Output file for benchmark report"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🧮 Math Routing Agent - JEE Benchmark Evaluation")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output File: {args.output}")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please fix the issues and try again.")
        sys.exit(1)
    
    try:
        # Run the benchmark
        logger.info("🚀 Starting JEE Benchmark Evaluation...")
        metrics = await run_jee_benchmark()
        
        # Save results with custom filename
        if args.output != "jee_benchmark_report.json":
            import json
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"📊 Benchmark report saved to {args.output}")
        
        # Print final summary
        summary = metrics["benchmark_summary"]
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Overall Accuracy: {summary['overall_accuracy']:.2%}")
        print(f"   Total Problems: {summary['total_problems']}")
        print(f"   Correct Answers: {summary['correct_answers']}")
        print(f"   Avg Response Time: {summary['avg_response_time']:.2f}s")
        
        # Performance recommendations
        if summary['overall_accuracy'] >= 0.8:
            print("🎉 Excellent performance! Your agent is performing well on JEE problems.")
        elif summary['overall_accuracy'] >= 0.6:
            print("👍 Good performance! Consider fine-tuning for better accuracy.")
        else:
            print("⚠️ Performance needs improvement. Consider:")
            print("   - Enhancing knowledge base coverage")
            print("   - Improving DSPy Chain-of-Thought prompts")
            print("   - Adding more training examples")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Benchmark evaluation failed: {e}")
        print(f"\n❌ Benchmark failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())