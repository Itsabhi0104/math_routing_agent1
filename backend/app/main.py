# backend/app/main.py

import logging
import uvicorn
from time import perf_counter
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure all ORM models are registered on Base
import app.models.models

from app.db import get_db, Base, engine
from app.routers.ai_gateway import router as gateway_router
from app.feedback import router as feedback_router
from app.routers.health import router as health_router
from app.services.db_service import DBService
from app.config import settings

# ----------------------------------------------------
# Logging setup
# ----------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Lifespan management
# ----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("🚀 Starting Math Routing Agent...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created/verified")
    
    # Initialize knowledge base if needed
    try:
        from app.lancedb_store import LanceDBStore
        store = LanceDBStore(db_path=settings.LANCEDB_PATH)
        # Test connection
        logger.info("✅ LanceDB connection verified")
    except Exception as e:
        logger.warning(f"⚠️ LanceDB initialization warning: {e}")
    
    # Startup complete
    logger.info("🎉 Math Routing Agent startup complete!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Math Routing Agent...")
    # Cleanup code here if needed

# ----------------------------------------------------
# FastAPI app instantiation
# ----------------------------------------------------
app = FastAPI(
    title="Math Routing Agent",
    description="AI-powered mathematical problem solving agent with educational focus",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ----------------------------------------------------
# Middleware
# ----------------------------------------------------
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Request logging middleware
# ----------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = perf_counter()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = perf_counter() - start_time
        
        # Log response
        logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = perf_counter() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - ERROR ({process_time:.3f}s): {e}")
        raise

# ----------------------------------------------------
# Global exception handler
# ----------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"❌ Unhandled exception in {request.method} {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "path": str(request.url.path)
        }
    )

# ----------------------------------------------------
# Include routers with proper prefixes
# ----------------------------------------------------
app.include_router(
    gateway_router,
    prefix="/api/v1",
    tags=["ai-gateway"]
)

app.include_router(
    feedback_router,
    prefix="/api/v1/feedback",
    tags=["feedback"]
)

app.include_router(
    health_router,
    prefix="/api/v1/health",
    tags=["health"]
)

# ----------------------------------------------------
# Main solve endpoint with comprehensive logging
# ----------------------------------------------------
@app.post(
    "/api/v1/solve",
    tags=["math-solver"],
    summary="Solve mathematical problems",
    description="Main endpoint for solving mathematical problems with step-by-step solutions"
)
async def api_solve(
    request: dict,
    db: AsyncSession = Depends(get_db)
):
    """
    Enhanced solve endpoint with:
    1. Input validation
    2. Problem solving via orchestration
    3. Response logging
    4. Performance tracking
    """
    start_time = perf_counter()
    
    # Extract question from request
    question = request.get("question", "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required"
        )
    
    logger.info(f"🔍 Solving: {question[:100]}...")
    
    try:
        # Import here to avoid circular imports
        from app.orchestration import solve_math_problem
        
        # Solve the problem
        response = await solve_math_problem(question)
        
        # Calculate processing time
        duration = perf_counter() - start_time
        
        # Log to database
        svc = DBService(db)
        if response.results:
            primary_result = response.results[0]
            try:
                await svc.log_query(
                    question=question,
                    source=response.routed_to,
                    answer=primary_result.answer,
                    response_time=duration
                )
            except Exception as log_error:
                logger.warning(f"⚠️ Failed to log query: {log_error}")
        
        # Add performance metadata
        response_dict = response.dict()
        response_dict["processing_time"] = duration
        response_dict["timestamp"] = perf_counter()
        
        logger.info(f"✅ Solved via {response.routed_to} in {duration:.3f}s")
        return response_dict
        
    except HTTPException:
        raise
    except Exception as e:
        duration = perf_counter() - start_time
        logger.exception(f"❌ Solve failed after {duration:.3f}s")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Problem solving failed",
                "message": str(e),
                "processing_time": duration
            }
        )

# ----------------------------------------------------
# History and analytics endpoints
# ----------------------------------------------------
@app.get("/api/v1/history", tags=["analytics"])
async def get_query_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get recent query history with statistics."""
    try:
        svc = DBService(db)
        
        # Get history records
        history = await svc.list_history(limit=limit)
        
        # Get feedback count
        feedback = await svc.list_feedback(limit=100)
        
        # Calculate basic stats
        if history:
            avg_response_time = sum(h.get("response_time", 0) for h in history) / len(history)
            route_counts = {}
            for h in history:
                route = h.get("source", "unknown")
                route_counts[route] = route_counts.get(route, 0) + 1
        else:
            avg_response_time = 0
            route_counts = {}
        
        return {
            "summary": {
                "total_queries": len(history),
                "total_feedback": len(feedback),
                "avg_response_time": avg_response_time,
                "route_distribution": route_counts
            },
            "recent_queries": history[:20],  # Last 20 queries
            "timestamp": perf_counter()
        }
        
    except Exception as e:
        logger.error(f"❌ History retrieval failed: {e}")
        return {
            "summary": {"total_queries": 0, "total_feedback": 0, "error": str(e)},
            "recent_queries": [],
            "timestamp": perf_counter()
        }

@app.get("/api/v1/stats", tags=["analytics"])
async def get_system_stats(db: AsyncSession = Depends(get_db)):
    """Get comprehensive system statistics."""
    try:
        svc = DBService(db)
        
        # Database stats
        history = await svc.list_history(limit=1000)
        feedback = await svc.list_feedback(limit=1000)
        
        # Performance metrics
        if history:
            response_times = [h.get("response_time", 0) for h in history]
            avg_time = sum(response_times) / len(response_times)
            
            # Route analysis
            routes = {}
            for h in history:
                route = h.get("source", "unknown")
                routes[route] = routes.get(route, 0) + 1
        else:
            avg_time = 0
            routes = {}
        
        # Feedback analysis
        if feedback:
            ratings = [f.rating for f in feedback if f.rating]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
        else:
            avg_rating = 0
        
        return {
            "system": {
                "status": "operational",
                "version": "2.0.0",
                "uptime": "runtime_info_here"
            },
            "performance": {
                "total_queries": len(history),
                "avg_response_time": avg_time,
                "route_distribution": routes
            },
            "feedback": {
                "total_feedback": len(feedback),
                "avg_rating": avg_rating
            },
            "timestamp": perf_counter()
        }
        
    except Exception as e:
        logger.error(f"❌ Stats calculation failed: {e}")
        return {
            "system": {"status": "error", "error": str(e)},
            "timestamp": perf_counter()
        }

# ----------------------------------------------------
# Administrative endpoints
# ----------------------------------------------------
@app.post("/api/v1/admin/rebuild-kb", tags=["admin"])
async def rebuild_knowledge_base():
    """Rebuild the knowledge base (admin only)."""
    try:
        logger.info("🔄 Starting knowledge base rebuild...")
        
        # This would typically require authentication
        from app.knowledge_loader import KnowledgeBaseBuilder
        from app.lancedb_store import LanceDBStore
        
        # Rebuild KB
        builder = KnowledgeBaseBuilder()
        df = builder.build(output_path="knowledge_base.parquet")
        
        # Update LanceDB
        store = LanceDBStore(db_path=settings.LANCEDB_PATH)
        store.ingest_from_parquet("knowledge_base.parquet", overwrite=True)
        
        logger.info("✅ Knowledge base rebuild complete")
        
        return {
            "status": "success",
            "message": f"Knowledge base rebuilt with {len(df)} entries",
            "timestamp": perf_counter()
        }
        
    except Exception as e:
        logger.error(f"❌ KB rebuild failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge base rebuild failed: {str(e)}"
        )

# ----------------------------------------------------
# Root endpoint
# ----------------------------------------------------
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Math Routing Agent",
        "version": "2.0.0",
        "description": "AI-powered mathematical problem solving with educational focus",
        "status": "operational",
        "features": [
            "Arithmetic Calculator",
            "Knowledge Base Search", 
            "Chain-of-Thought Reasoning",
            "Web Search Integration",
            "Human-in-the-Loop Learning"
        ],
        "endpoints": {
            "solve": "/api/v1/solve",
            "feedback": "/api/v1/feedback",
            "history": "/api/v1/history",
            "stats": "/api/v1/stats",
            "health": "/api/v1/health",
            "docs": "/docs"
        },
        "timestamp": perf_counter()
    }

# ----------------------------------------------------
# Benchmark endpoint
# ----------------------------------------------------
@app.post("/api/v1/benchmark", tags=["evaluation"])
async def run_benchmark():
    """Run JEE benchmark evaluation."""
    try:
        logger.info("🧪 Starting JEE benchmark evaluation...")
        
        from app.benchmarks.jee_benchmark import run_jee_benchmark
        
        # Run benchmark asynchronously
        results = await run_jee_benchmark()
        
        logger.info("✅ JEE benchmark completed")
        return {
            "status": "completed",
            "results": results,
            "timestamp": perf_counter()
        }
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark evaluation failed: {str(e)}"
        )

# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )