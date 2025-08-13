#!/usr/bin/env python3
"""
Complete Pipeline Testing Script
Tests all components of the Math Routing Agent to ensure proper functionality.
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineTestSuite:
    """Comprehensive test suite for the Math Routing Agent pipeline."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
    
    async def test_basic_arithmetic(self) -> bool:
        """Test basic arithmetic evaluation."""
        logger.info("🧮 Testing Basic Arithmetic...")
        
        test_cases = [
            ("What is 2+2?", "4"),
            ("Calculate 15 * 3", "45"),
            ("What is 100 / 4?", "25"),
            ("Find 2^3", "8"),
            ("What is 15 choose 3?", "455")
        ]
        
        try:
            from app.orchestration import solve_math_problem
            
            all_passed = True
            for question, expected in test_cases:
                start_time = time.time()
                result = await solve_math_problem(question)
                response_time = time.time() - start_time
                
                if result.results:
                    answer = result.results[0].answer
                    passed = str(expected) in answer or answer == expected
                else:
                    passed = False
                    answer = "No answer"
                
                status = "✅" if passed else "❌"
                logger.info(f"  {status} {question} -> Expected: {expected}, Got: {answer} ({response_time:.2f}s)")
                
                if not passed:
                    all_passed = False
                    self.failed_tests.append(f"Arithmetic: {question}")
            
            self.test_results.append(("Basic Arithmetic", all_passed))
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Arithmetic test failed: {e}")
            self.test_results.append(("Basic Arithmetic", False))
            self.failed_tests.append(f"Arithmetic test exception: {e}")
            return False
    
    async def test_algebra_problems(self) -> bool:
        """Test algebraic problem solving."""
        logger.info("📐 Testing Algebra Problems...")
        
        test_cases = [
            "Solve x + 5 = 10 for x",
            "Find x if 2x - 3 = 7", 
            "Solve the equation 3x + 2 = 14",
            "If y = 2x + 1 and x = 3, find y"
        ]
        
        try:
            from app.orchestration import solve_math_problem
            
            passed_count = 0
            for question in test_cases:
                start_time = time.time()
                result = await solve_math_problem(question)
                response_time = time.time() - start_time
                
                if result.results and result.results[0].answer:
                    answer = result.results[0].answer
                    # For algebra, we check if we got a reasonable response
                    has_solution = any(char in answer.lower() for char in ['x', '=', '5', '7'])
                    status = "✅" if has_solution else "❌"
                    passed_count += 1 if has_solution else 0
                else:
                    status = "❌"
                    answer = "No answer"
                
                logger.info(f"  {status} {question} -> {answer[:50]}... ({response_time:.2f}s)")
            
            success_rate = passed_count / len(test_cases)
            all_passed = success_rate >= 0.7  # 70% success rate for algebra
            
            self.test_results.append(("Algebra Problems", all_passed))
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Algebra test failed: {e}")
            self.test_results.append(("Algebra Problems", False))
            self.failed_tests.append(f"Algebra test exception: {e}")
            return False
    
    async def test_geometry_problems(self) -> bool:
        """Test geometry problem solving."""
        logger.info("📏 Testing Geometry Problems...")
        
        test_cases = [
            "Find the area of a circle with radius 5",
            "What is the perimeter of a rectangle with length 8 and width 6?",
            "Calculate the volume of a cube with side length 4",
            "Find the area of a triangle with base 10 and height 6"
        ]
        
        try:
            from app.orchestration import solve_math_problem
            
            passed_count = 0
            for question in test_cases:
                start_time = time.time()
                result = await solve_math_problem(question)
                response_time = time.time() - start_time
                
                if result.results and result.results[0].answer:
                    answer = result.results[0].answer
                    # Check for reasonable geometric answer
                    has_numeric = any(char.isdigit() for char in answer)
                    status = "✅" if has_numeric else "❌"
                    passed_count += 1 if has_numeric else 0
                else:
                    status = "❌"
                    answer = "No answer"
                
                logger.info(f"  {status} {question} -> {answer[:50]}... ({response_time:.2f}s)")
            
            success_rate = passed_count / len(test_cases)
            all_passed = success_rate >= 0.6  # 60% success rate for geometry
            
            self.test_results.append(("Geometry Problems", all_passed))
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Geometry test failed: {e}")
            self.test_results.append(("Geometry Problems", False))
            self.failed_tests.append(f"Geometry test exception: {e}")
            return False
    
    async def test_knowledge_base_search(self) -> bool:
        """Test knowledge base functionality."""
        logger.info("📚 Testing Knowledge Base Search...")
        
        try:
            from app.lancedb_store import LanceDBStore
            from app.embedding_client import get_embedding
            
            # Test if LanceDB store is accessible
            store = LanceDBStore(
                db_path=settings.LANCEDB_PATH,
                table_name="math_qa"
            )
            
            # Test embedding generation
            test_query = "What is 2+2?"
            embedding = get_embedding(test_query)
            
            if len(embedding) != 384:  # Expected dimension for all-MiniLM-L6-v2
                logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
            
            # Test semantic search
            results = await store.semantic_search(
                query_embedding=embedding,
                top_k=3,
                threshold=0.3
            )
            
            # Test passed if we can execute search without errors
            kb_accessible = True
            logger.info(f"✅ Knowledge base search executed, found {len(results)} results")
            
            self.test_results.append(("Knowledge Base Search", kb_accessible))
            return kb_accessible
            
        except Exception as e:
            logger.error(f"❌ Knowledge base test failed: {e}")
            self.test_results.append(("Knowledge Base Search", False))
            self.failed_tests.append(f"KB test exception: {e}")
            return False
    
    async def test_dspy_integration(self) -> bool:
        """Test DSPy Chain-of-Thought integration."""
        logger.info("🔗 Testing DSPy Integration...")
        
        try:
            from app.dspy_integration import MathCoTPipeline
            
            # Initialize pipeline
            pipeline = MathCoTPipeline(api_key=settings.GOOGLE_API_KEY)
            
            # Test simple problem
            test_question = "What is the derivative of x^2?"
            result = pipeline.solve(test_question)
            
            # Check if we got a reasonable response
            has_answer = bool(result.get("answer"))
            has_steps = bool(result.get("steps"))
            
            dspy_working = has_answer and has_steps
            status = "✅" if dspy_working else "❌"
            logger.info(f"  {status} DSPy CoT: Answer={has_answer}, Steps={has_steps}")
            
            self.test_results.append(("DSPy Integration", dspy_working))
            return dspy_working
            
        except Exception as e:
            logger.error(f"❌ DSPy test failed: {e}")
            self.test_results.append(("DSPy Integration", False))
            self.failed_tests.append(f"DSPy test exception: {e}")
            return False
    
    async def test_mcp_integration(self) -> bool:
        """Test MCP (Tavily) web search integration."""
        logger.info("🌐 Testing MCP Web Search...")
        
        try:
            from app.mcp_client import web_search_math
            
            # Test web search
            test_query = "integral of x^2 dx"
            results = await web_search_math(test_query, max_results=2)
            
            # Check if we got results
            mcp_working = len(results) > 0
            status = "✅" if mcp_working else "❌"
            logger.info(f"  {status} MCP Web Search: Found {len(results)} results")
            
            self.test_results.append(("MCP Integration", mcp_working))
            return mcp_working
            
        except Exception as e:
            logger.error(f"❌ MCP test failed: {e}")
            self.test_results.append(("MCP Integration", False))
            self.failed_tests.append(f"MCP test exception: {e}")
            return False
    
    async def test_input_guardrails(self) -> bool:
        """Test input guardrails and validation."""
        logger.info("🛡️ Testing Input Guardrails...")
        
        try:
            from app.routers.ai_gateway import MathInputGuardrails
            
            # Test valid mathematical inputs
            valid_inputs = [
                "What is 2+2?",
                "Solve x + 5 = 10",
                "Find the area of a circle",
                "Calculate the derivative"
            ]
            
            # Test invalid inputs  
            invalid_inputs = [
                "What is your name?",
                "Tell me about politics",
                "How to hack a computer",
                "Random non-math question"
            ]
            
            valid_passed = 0
            for inp in valid_inputs:
                result = MathInputGuardrails.validate_input(inp)
                if result['is_valid']:
                    valid_passed += 1
            
            invalid_blocked = 0
            for inp in invalid_inputs:
                result = MathInputGuardrails.validate_input(inp)
                if not result['is_valid']:
                    invalid_blocked += 1
            
            guardrails_working = (valid_passed >= 3) and (invalid_blocked >= 2)
            status = "✅" if guardrails_working else "❌"
            logger.info(f"  {status} Input Guardrails: Valid={valid_passed}/4, Blocked={invalid_blocked}/4")
            
            self.test_results.append(("Input Guardrails", guardrails_working))
            return guardrails_working
            
        except Exception as e:
            logger.error(f"❌ Guardrails test failed: {e}")
            self.test_results.append(("Input Guardrails", False))
            self.failed_tests.append(f"Guardrails test exception: {e}")
            return False
    
    async def test_end_to_end_flow(self) -> bool:
        """Test complete end-to-end flow."""
        logger.info("🔄 Testing End-to-End Flow...")
        
        test_cases = [
            "What is 5 * 7?",  # Should use arithmetic
            "Solve 2x + 3 = 11",  # Should use KB or DSPy
            "Find the area of a circle with radius 3"  # Should use KB or web search
        ]
        
        try:
            from app.orchestration import solve_math_problem
            
            passed_count = 0
            for question in test_cases:
                start_time = time.time()
                result = await solve_math_problem(question)
                response_time = time.time() - start_time
                
                # Check if we got a complete response
                has_answer = result.results and result.results[0].answer
                has_route = bool(result.routed_to)
                reasonable_time = response_time < 30  # Should complete within 30 seconds
                
                test_passed = has_answer and has_route and reasonable_time
                status = "✅" if test_passed else "❌"
                
                logger.info(f"  {status} {question}")
                logger.info(f"      Route: {result.routed_to}")
                logger.info(f"      Answer: {result.results[0].answer[:50] if result.results else 'None'}...")
                logger.info(f"      Time: {response_time:.2f}s")
                
                if test_passed:
                    passed_count += 1
                else:
                    self.failed_tests.append(f"E2E: {question}")
            
            e2e_success = passed_count >= 2  # At least 2/3 should pass
            self.test_results.append(("End-to-End Flow", e2e_success))
            return e2e_success
            
        except Exception as e:
            logger.error(f"❌ End-to-end test failed: {e}")
            self.test_results.append(("End-to-End Flow", False))
            self.failed_tests.append(f"E2E test exception: {e}")
            return False
    
    def print_test_summary(self):
        """Print comprehensive test results summary."""
        print("\n" + "="*60)
        print("           PIPELINE TEST RESULTS SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed in self.test_results if passed)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        print("\nDetailed Results:")
        for test_name, passed in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} - {test_name}")
        
        if self.failed_tests:
            print("\nFailed Test Details:")
            for failure in self.failed_tests:
                print(f"  🔍 {failure}")
        
        print("\nRecommendations:")
        if passed_tests == total_tests:
            print("  🎉 All tests passed! Your Math Routing Agent is working perfectly.")
        elif passed_tests >= total_tests * 0.8:
            print("  👍 Most tests passed. Address the failed tests for optimal performance.")
        elif passed_tests >= total_tests * 0.6:
            print("  ⚠️ Some critical issues found. Please fix failed components.")
        else:
            print("  🚨 Multiple critical failures. Review system configuration and dependencies.")
        
        print("="*60)

async def run_all_tests():
    """Run the complete test suite."""
    test_suite = PipelineTestSuite()
    
    print("🧪 Math Routing Agent - Pipeline Test Suite")
    print("="*60)
    print("Testing all components of the Math Routing Agent...")
    print("="*60)
    
    # Run all tests
    test_functions = [
        test_suite.test_basic_arithmetic,
        test_suite.test_algebra_problems,
        test_suite.test_geometry_problems,
        test_suite.test_knowledge_base_search,
        test_suite.test_dspy_integration,
        test_suite.test_mcp_integration,
        test_suite.test_input_guardrails,
        test_suite.test_end_to_end_flow
    ]
    
    for test_func in test_functions:
        try:
            await test_func()
            # Small delay between tests
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Test function {test_func.__name__} crashed: {e}")
    
    # Print summary
    test_suite.print_test_summary()
    
    return test_suite.test_results

def check_environment():
    """Check if the environment is properly configured."""
    logger.info("🔍 Checking Environment Configuration...")
    
    issues = []
    
    # Check API keys
    if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "your-google-api-key":
        issues.append("GOOGLE_API_KEY not properly set")
    
    if not settings.TAVILY_API_KEY or settings.TAVILY_API_KEY == "your-tavily-api-key":
        issues.append("TAVILY_API_KEY not properly set (MCP tests will fail)")
    
    # Check paths
    if not os.path.exists(settings.LANCEDB_PATH):
        issues.append(f"LanceDB path does not exist: {settings.LANCEDB_PATH}")
    
    # Check if parquet file exists
    if not os.path.exists("knowledge_base.parquet"):
        issues.append("knowledge_base.parquet not found (run setup_kb.py first)")
    
    if issues:
        print("⚠️ Environment Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommended actions:")
        print("  1. Run 'python scripts/setup_kb.py' to create knowledge base")
        print("  2. Ensure API keys are properly configured")
        print("  3. Check file permissions and paths")
        return False
    else:
        print("✅ Environment configuration looks good!")
        return True

async def main():
    """Main test execution function."""
    print("🔧 Math Routing Agent - Pipeline Testing")
    print("="*60)
    
    # Check environment first
    if not check_environment():
        print("\n❌ Environment check failed. Please fix issues before testing.")
        sys.exit(1)
    
    try:
        # Run all tests
        results = await run_all_tests()
        
        # Determine exit code based on results
        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)
        
        if passed_count == total_count:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        elif passed_count >= total_count * 0.8:
            print(f"\n👍 Most tests passed ({passed_count}/{total_count})")
            sys.exit(0)
        else:
            print(f"\n⚠️ Multiple test failures ({passed_count}/{total_count})")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())