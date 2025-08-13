# backend/app/mcp_client.py

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
import json

import httpx
from cachetools import TTLCache
from pydantic import BaseModel, Field, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

# ————————————————————————————————————————————
# Pydantic Models
# ————————————————————————————————————————————

class MCPRequest(BaseModel):
    """Request model for MCP search."""
    query: str = Field(..., description="Mathematical query for web search")
    max_results: int = Field(5, ge=1, le=20, description="Maximum search results")
    include_domains: Optional[List[str]] = Field(None, description="Domains to include")
    exclude_domains: Optional[List[str]] = Field(None, description="Domains to exclude")

class MCPResult(BaseModel):
    """Individual search result from MCP."""
    title: str
    snippet: str
    url: str
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    domain: Optional[str] = None
    published_date: Optional[str] = None

class MCPResponse(BaseModel):
    """Complete response from MCP search."""
    query: str
    results: List[MCPResult]
    total_results: int
    search_time: float
    status: str = "success"

# ————————————————————————————————————————————
# Caching Setup
# ————————————————————————————————————————————
# Cache search results for 10 minutes to reduce API calls
mcp_cache: TTLCache = TTLCache(maxsize=200, ttl=600)

# ————————————————————————————————————————————
# Math Content Filters
# ————————————————————————————————————————————

class MathContentFilter:
    """Filter and validate mathematical content from search results."""
    
    # Mathematical indicators
    MATH_KEYWORDS = [
        'equation', 'formula', 'calculate', 'solve', 'derivative', 'integral',
        'geometry', 'algebra', 'calculus', 'trigonometry', 'mathematics',
        'theorem', 'proof', 'solution', 'problem', 'answer'
    ]
    
    MATH_SYMBOLS = ['=', '+', '-', '*', '/', '^', '√', '∫', '∑', 'π', '∞', '≤', '≥']
    
    # Domains likely to contain quality math content
    TRUSTED_MATH_DOMAINS = [
        'mathworld.wolfram.com',
        'khanacademy.org',
        'wikipedia.org',
        'mathpages.com',
        'cut-the-knot.org',
        'artofproblemsolving.com',
        'brilliant.org',
        'stackexchange.com',
        'mathoverflow.net'
    ]
    
    @classmethod
    def is_math_related(cls, text: str) -> bool:
        """Check if text contains mathematical content."""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check for math keywords
        keyword_score = sum(1 for keyword in cls.MATH_KEYWORDS if keyword in text_lower)
        
        # Check for math symbols
        symbol_score = sum(1 for symbol in cls.MATH_SYMBOLS if symbol in text)
        
        # Check for numbers and basic patterns
        number_patterns = re.findall(r'\d+', text)
        equation_patterns = re.findall(r'\w+\s*=\s*\w+', text)
        
        # Calculate overall math score
        total_score = keyword_score + symbol_score + len(number_patterns) * 0.1 + len(equation_patterns) * 2
        
        return total_score >= 1.0
    
    @classmethod
    def calculate_math_relevance(cls, result: Dict[str, Any]) -> float:
        """Calculate relevance score for mathematical content."""
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        url = result.get('url', '')
        
        relevance = 0.0
        
        # Title scoring (higher weight)
        if cls.is_math_related(title):
            relevance += 0.4
        
        # Snippet scoring
        if cls.is_math_related(snippet):
            relevance += 0.3
        
        # Domain scoring
        domain = cls._extract_domain(url)
        if domain in cls.TRUSTED_MATH_DOMAINS:
            relevance += 0.2
        elif 'math' in domain or 'edu' in domain:
            relevance += 0.1
        
        # Length and quality indicators
        if len(snippet) > 100:  # Substantial content
            relevance += 0.05
        
        if 'step' in snippet.lower() or 'solution' in snippet.lower():
            relevance += 0.1
        
        return min(1.0, relevance)
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

# ————————————————————————————————————————————
# MCP Client Implementation
# ————————————————————————————————————————————

class MCPClient:
    """
    Enhanced async client for Tavily MCP-based mathematical web search.
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or settings.TAVILY_API_KEY
        self.base_url = base_url or settings.MCP_URL
        self.timeout = httpx.Timeout(settings.MCP_TIMEOUT, connect=10.0)
        self.content_filter = MathContentFilter()
        
        # Create async HTTP client
        self._client = None
        
        logger.info(f"Initialized MCPClient with timeout={settings.MCP_TIMEOUT}s")
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={'User-Agent': 'MathRoutingAgent/2.0'}
            )
        return self._client
    
    async def search(self, request: MCPRequest) -> MCPResponse:
        """
        Perform MCP web search for mathematical content.
        
        Args:
            request: Search request parameters
            
        Returns:
            Structured search response
        """
        # Check cache first
        cache_key = f"{request.query}:{request.max_results}"
        cached_result = mcp_cache.get(cache_key)
        if cached_result:
            logger.info(f"📦 Cache hit for query: {request.query}")
            return cached_result
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(request)
            
            logger.info(f"🔍 MCP Search: {request.query}")
            
            # Execute search
            response = await self.client.get(search_url)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            raw_results = data.get('results', [])
            
            # Process and filter results
            processed_results = []
            for result in raw_results[:request.max_results]:
                try:
                    # Create result object
                    mcp_result = MCPResult(
                        title=result.get('title', ''),
                        snippet=result.get('snippet', ''),
                        url=result.get('url', ''),
                        domain=self.content_filter._extract_domain(result.get('url', '')),
                        published_date=result.get('published_date')
                    )
                    
                    # Calculate mathematical relevance
                    relevance = self.content_filter.calculate_math_relevance(result)
                    mcp_result.relevance = relevance
                    
                    # Only include if it's math-related
                    if relevance > 0.1:  # Minimum threshold
                        processed_results.append(mcp_result)
                    else:
                        logger.debug(f"Filtered non-math result: {mcp_result.title}")
                        
                except ValidationError as e:
                    logger.warning(f"Invalid result format: {e}")
                    continue
            
            # Sort by relevance
            processed_results.sort(key=lambda x: x.relevance, reverse=True)
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            # Create response
            mcp_response = MCPResponse(
                query=request.query,
                results=processed_results,
                total_results=len(processed_results),
                search_time=search_time
            )
            
            # Cache successful results
            mcp_cache[cache_key] = mcp_response
            
            logger.info(f"✅ MCP search completed: {len(processed_results)} math results in {search_time:.2f}s")
            return mcp_response
            
        except httpx.TimeoutException:
            logger.error(f"⏰ MCP search timeout for query: {request.query}")
            return MCPResponse(
                query=request.query,
                results=[],
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                status="timeout"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ MCP HTTP error {e.response.status_code}: {e}")
            return MCPResponse(
                query=request.query,
                results=[],
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                status=f"http_error_{e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"❌ MCP search failed: {e}")
            return MCPResponse(
                query=request.query,
                results=[],
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                status="error"
            )
    
    def _build_search_url(self, request: MCPRequest) -> str:
        """Build the search URL with parameters."""
        params = {
            'query': self._enhance_math_query(request.query),
            'max_results': request.max_results,
            'search_depth': 'basic',
            'include_answer': True,
            'include_raw_content': False
        }
        
        # Add domain filters if specified
        if request.include_domains:
            params['include_domains'] = ','.join(request.include_domains)
        
        if request.exclude_domains:
            params['exclude_domains'] = ','.join(request.exclude_domains)
        
        # Build full URL
        base_url = self.base_url.rstrip('/')
        if not base_url.endswith('/search'):
            base_url += '/search'
        
        # Add API key
        if 'tavilyApiKey' not in base_url:
            separator = '&' if '?' in base_url else '?'
            base_url += f"{separator}tavilyApiKey={self.api_key}"
        
        # Add search parameters
        query_string = urlencode(params)
        return f"{base_url}&{query_string}"
    
    def _enhance_math_query(self, query: str) -> str:
        """Enhance query with mathematical context."""
        enhanced_query = query.strip()
        
        # Add mathematical context if not present
        math_indicators = ['math', 'mathematical', 'equation', 'formula', 'calculate', 'solve']
        if not any(indicator in enhanced_query.lower() for indicator in math_indicators):
            enhanced_query = f"mathematics {enhanced_query}"
        
        # Add step-by-step context for educational value
        if 'step' not in enhanced_query.lower() and 'how' not in enhanced_query.lower():
            enhanced_query += " step by step solution"
        
        return enhanced_query
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

# ————————————————————————————————————————————
# High-level search functions
# ————————————————————————————————————————————

async def web_search_math(
    question: str, 
    max_results: int = 5,
    include_domains: Optional[List[str]] = None
) -> List[MCPResult]:
    """
    High-level function for mathematical web search.
    
    Args:
        question: Mathematical question to search for
        max_results: Maximum number of results
        include_domains: Optional list of domains to prioritize
        
    Returns:
        List of relevant mathematical search results
    """
    if not question or len(question.strip()) < 3:
        logger.warning("Query too short for web search")
        return []
    
    # Add trusted math domains to include list
    trusted_domains = [
        'mathworld.wolfram.com',
        'khanacademy.org', 
        'brilliant.org',
        'wikipedia.org'
    ]
    
    if include_domains:
        include_domains.extend(trusted_domains)
    else:
        include_domains = trusted_domains
    
    client = MCPClient()
    
    try:
        request = MCPRequest(
            query=question,
            max_results=max_results,
            include_domains=include_domains
        )
        
        response = await client.search(request)
        
        if response.status == "success":
            logger.info(f"🌐 Web search found {len(response.results)} mathematical results")
            return response.results
        else:
            logger.warning(f"⚠️ Web search completed with status: {response.status}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Web search failed: {e}")
        return []
    finally:
        await client.close()

async def search_math_concepts(concept: str) -> List[MCPResult]:
    """
    Search for explanations of mathematical concepts.
    
    Args:
        concept: Mathematical concept to search for
        
    Returns:
        List of educational resources about the concept
    """
    enhanced_query = f"what is {concept} mathematics explanation examples"
    
    # Prioritize educational domains
    educational_domains = [
        'khanacademy.org',
        'brilliant.org',
        'mathsisfun.com',
        'purplemath.com'
    ]
    
    return await web_search_math(
        enhanced_query, 
        max_results=3,
        include_domains=educational_domains
    )

async def search_problem_solutions(problem: str) -> List[MCPResult]:
    """
    Search for solutions to specific mathematical problems.
    
    Args:
        problem: Mathematical problem to find solutions for
        
    Returns:
        List of solution resources
    """
    enhanced_query = f"solve {problem} step by step solution"
    
    # Prioritize solution-oriented domains
    solution_domains = [
        'mathway.com',
        'symbolab.com',
        'wolframalpha.com',
        'stackexchange.com'
    ]
    
    return await web_search_math(
        enhanced_query,
        max_results=3, 
        include_domains=solution_domains
    )

# ————————————————————————————————————————————
# Cache management
# ————————————————————————————————————————————

def get_cache_stats() -> Dict[str, Any]:
    """Get MCP cache statistics."""
    return {
        "cache_size": len(mcp_cache),
        "max_size": mcp_cache.maxsize,
        "ttl": mcp_cache.ttl
    }

def clear_mcp_cache() -> None:
    """Clear the MCP search cache."""
    mcp_cache.clear()
    logger.info("🧹 MCP cache cleared")

# ————————————————————————————————————————————
# Testing and validation
# ————————————————————————————————————————————

async def test_mcp_connection() -> bool:
    """Test MCP connection and basic functionality."""
    try:
        logger.info("🧪 Testing MCP connection...")
        
        results = await web_search_math("what is 2+2", max_results=1)
        
        if results:
            logger.info("✅ MCP connection test passed")
            return True
        else:
            logger.warning("⚠️ MCP connection test returned no results")
            return False
            
    except Exception as e:
        logger.error(f"❌ MCP connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the MCP client
    async def main():
        # Test basic search
        results = await web_search_math("derivative of x^2")
        print(f"Found {len(results)} results")
        
        for result in results[:2]:
            print(f"Title: {result.title}")
            print(f"Snippet: {result.snippet[:100]}...")
            print(f"Relevance: {result.relevance:.2f}")
            print("-" * 50)
    
    asyncio.run(main())