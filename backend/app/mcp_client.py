# backend/app/mcp_client.py

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from cachetools import TTLCache, cached
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
import sympy as sp

logger = logging.getLogger(__name__)

# ————————————————————————————————————————————
# Pydantic Models
# ————————————————————————————————————————————

class MCPRequest(BaseModel):
    query: str = Field(..., description="Mathematical query for web search")
    max_results: int = Field(5, ge=1, le=20, description="Max search hits")


class MCPResult(BaseModel):
    title: str
    snippet: str
    url: str
    relevance: float


# ————————————————————————————————————————————
# Caching Setup
# ————————————————————————————————————————————
# Cache up to 100 distinct queries, 10-minute TTL
mcp_cache: TTLCache = TTLCache(maxsize=100, ttl=600)


# ————————————————————————————————————————————
# MCP Client
# ————————————————————————————————————————————

class MCPClient:
    """
    Async client for Tavily MCP-based web search.
    """

    def __init__(self, api_key: str, base_url: str):
        self.base_url = f"{base_url}?tavilyApiKey={api_key}"
        self.timeout = httpx.Timeout(15.0, connect=5.0)
        self._client = httpx.AsyncClient(timeout=self.timeout)
        logger.info("Initialized MCPClient for %s", self.base_url)

    @cached(mcp_cache)
    async def search(self, req: MCPRequest) -> List[MCPResult]:
        """
        Perform an MCP web search for mathematical queries,
        with caching to prevent redundant calls.
        """
        params = {
            "model": "mcp-v1",
            "q": req.query,
            "max": req.max_results,
        }
        try:
            resp = await self._client.get(self.base_url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            hits = payload.get("results", [])
            results: List[MCPResult] = []
            for h in hits:
                try:
                    result = MCPResult(
                        title=h["title"],
                        snippet=h["snippet"],
                        url=h["url"],
                        relevance=float(h.get("score", 0.0)),
                    )
                    # filter: ensure snippet contains at least one math symbol
                    if any(sym in result.snippet for sym in ["=", "+", "-", "∫", "√"]):
                        results.append(result)
                    else:
                        logger.debug("Filtered non-math snippet: %s", result.snippet)
                except ValidationError as ve:
                    logger.warning("Invalid MCP hit format: %s", ve)
            logger.info("MCP search returned %d valid hits", len(results))
            return results

        except Exception as e:
            logger.exception("MCP search failed for query '%s'", req.query)
            raise

    async def close(self) -> None:
        await self._client.aclose()


# ————————————————————————————————————————————
# Search Pipeline
# ————————————————————————————————————————————

async def web_search_math(
    question: str, max_results: int = 5
) -> List[MCPResult]:
    """
    Full async pipeline: preprocess query, call MCPClient.search,
    validate results with SymPy, return structured MCPResult list.
    """
    # 1. Preprocess: strip whitespace, enforce minimal math content
    q = question.strip()
    if len(q) < 5:
        raise ValueError("Query too short for web search")

    # 2. Call MCP
    client = MCPClient(settings.TAVILY_API_KEY, settings.MCP_URL)
    try:
        req = MCPRequest(query=q, max_results=max_results)
        hits = await client.search(req)
    finally:
        await client.close()

    # 3. Validate with SymPy: ensure URLs/snippets parse or simplify
    validated: List[MCPResult] = []
    for hit in hits:
        # try to extract an equation snippet
        snippet = hit.snippet.split("…")[0]
        try:
            # attempt sympify; if fails, skip
            _ = sp.sympify(snippet)
            validated.append(hit)
        except Exception:
            logger.debug("SymPy failed to parse snippet: %s", snippet)

    logger.info("Web-search pipeline returned %d verified results", len(validated))
    return validated
