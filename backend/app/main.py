import logging
import uvicorn
from time import perf_counter
from typing import List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure all ORM models are registered on Base
import app.models.models

from app.db import get_db, Base, engine
from app.ai_gateway import QueryRequest, GatewayResponse, router as gateway_router
from app.feedback import router as feedback_router
from app.services.db_service import DBService

# ----------------------------------------------------
# Logging setup
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# FastAPI app instantiation
# ----------------------------------------------------
app = FastAPI(
    title="Math Routing Agent",
    version="1.0.0",
)

# ----------------------------------------------------
# Middleware
# ----------------------------------------------------
app.add_middleware(GZipMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Startup event: create DB tables
# ----------------------------------------------------
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created at startup")

# ----------------------------------------------------
# Health Check Endpoint (inline since health router seems problematic)
# ----------------------------------------------------
@app.get("/api/v1/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "Math Routing Agent", "version": "1.0.0"}

# ----------------------------------------------------
# Routers Registration
# ----------------------------------------------------
app.include_router(
    gateway_router,
    prefix="/api/v1",
    tags=["query"],
)

app.include_router(
    feedback_router,
    prefix="/api/v1/feedback",
    tags=["feedback"],
)

# ----------------------------------------------------
# Solve endpoint with logging
# ----------------------------------------------------
@app.post(
    "/api/v1/solve",
    response_model=GatewayResponse,
    status_code=status.HTTP_200_OK,
)
async def api_solve(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> GatewayResponse:
    """
    1) Solve the question via orchestration.
    2) Log the query into `query_history`.
    3) Return the GatewayResponse.
    """
    from app.orchestration import solve_math_problem

    svc = DBService(db)
    start = perf_counter()
    try:
        response = await solve_math_problem(request.question)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Unexpected error in solve_math_problem")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )
    duration = perf_counter() - start

    # Log into history: use the first answer as the canonical one
    if response.results:
        primary = response.results[0]
        try:
            await svc.log_query(
                question=request.question,
                source=response.routed_to,
                answer=primary.answer,
                response_time=duration,
            )
        except Exception:
            logger.exception("⚠️ Failed to log query history")
    else:
        logger.warning("No results returned from solve_math_problem")

    return response

# ----------------------------------------------------
# Summary counts endpoint
# ----------------------------------------------------
@app.get("/api/v1/history", status_code=status.HTTP_200_OK)
async def api_history(db: AsyncSession = Depends(get_db)):
    """
    Returns counts of recent queries & feedback.
    """
    try:
        svc = DBService(db)
        hist = await svc.list_history(limit=100)
        fb = await svc.list_feedback(limit=100)
        return {"total_queries": len(hist), "total_feedback": len(fb)}
    except Exception as e:
        logger.exception("Error fetching history")
        return {"total_queries": 0, "total_feedback": 0, "error": str(e)}

# ----------------------------------------------------
# Detailed history records endpoint
# ----------------------------------------------------
@app.get(
    "/api/v1/history/records",
    status_code=status.HTTP_200_OK,
)
async def api_history_records(
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Returns detailed list of recent queries (with IDs) for feedback.
    """
    try:
        svc = DBService(db)
        return await svc.list_history(limit=100)
    except Exception as e:
        logger.exception("Error fetching history records")
        return []

# ----------------------------------------------------
# Stats alias
# ----------------------------------------------------
@app.get("/api/v1/stats", status_code=status.HTTP_200_OK)
async def api_stats(db: AsyncSession = Depends(get_db)):
    return await api_history(db)

# ----------------------------------------------------
# Root endpoint for basic info
# ----------------------------------------------------
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "service": "Math Routing Agent",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/v1/health",
            "solve": "/api/v1/solve",
            "query": "/api/v1/query",
            "history": "/api/v1/history",
            "stats": "/api/v1/stats"
        }
    }

# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)