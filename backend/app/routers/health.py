# backend/app/routers/health.py

import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    Returns simple status without detailed checks.
    """
    return {
        "status": "healthy",
        "service": "Math Routing Agent",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Comprehensive health check with component status.
    Tests all major system components.
    """
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "service": "Math Routing Agent",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
        "performance": {},
        "configuration": {}
    }
    
    # Test database connection
    try:
        # Simple database query
        await db.execute("SELECT 1")
        health_status["components"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Test embedding service
    try:
        from app.embedding_client import get_embedding
        test_embedding = get_embedding("test")
        
        if test_embedding and len(test_embedding) == settings.EMBEDDING_DIMENSION:
            health_status["components"]["embedding_service"] = {
                "status": "healthy",
                "message": f"Embedding service operational (dim={len(test_embedding)})"
            }
        else:
            health_status["components"]["embedding_service"] = {
                "status": "unhealthy",
                "message": "Embedding service returned invalid results"
            }
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["components"]["embedding_service"] = {
            "status": "unhealthy",
            "message": f"Embedding service failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Test LanceDB
    try:
        from app.lancedb_store import LanceDBStore
        store = LanceDBStore()
        
        if store.table_exists():
            table_info = store.get_table_info()
            health_status["components"]["knowledge_base"] = {
                "status": "healthy",
                "message": f"Knowledge base operational ({table_info.get('total_records', 0)} records)"
            }
        else:
            health_status["components"]["knowledge_base"] = {
                "status": "degraded", 
                "message": "Knowledge base table not found"
            }
            
    except Exception as e:
        health_status["components"]["knowledge_base"] = {
            "status": "unhealthy",
            "message": f"Knowledge base check failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Test DSPy integration
    try:
        from app.dspy_integration import MathCoTPipeline
        
        # Just test initialization, not actual API call
        pipeline = MathCoTPipeline(api_key=settings.GOOGLE_API_KEY)
        
        health_status["components"]["dspy_pipeline"] = {
            "status": "healthy",
            "message": "DSPy pipeline initialized successfully"
        }
        
    except Exception as e:
        health_status["components"]["dspy_pipeline"] = {
            "status": "unhealthy",
            "message": f"DSPy pipeline initialization failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Test MCP client (lightweight test)
    try:
        from app.mcp_client import MCPClient
        
        # Just test client initialization
        client = MCPClient()
        await client.close()
        
        health_status["components"]["mcp_client"] = {
            "status": "healthy",
            "message": "MCP client initialized successfully"
        }
        
    except Exception as e:
        health_status["components"]["mcp_client"] = {
            "status": "degraded",
            "message": f"MCP client test failed: {str(e)}"
        }
    
    # Performance metrics
    response_time = time.time() - start_time
    health_status["performance"] = {
        "response_time_ms": round(response_time * 1000, 2),
        "status": "good" if response_time < 5.0 else "slow"
    }
    
    # Configuration validation
    config_validation = settings.validate_config()
    health_status["configuration"] = {
        "valid": config_validation["valid"],
        "issues": config_validation["issues"],
        "warnings": config_validation["warnings"]
    }
    
    if not config_validation["valid"]:
        health_status["status"] = "degraded"
    
    # Overall status determination
    unhealthy_components = [
        name for name, comp in health_status["components"].items()
        if comp["status"] == "unhealthy"
    ]
    
    if len(unhealthy_components) >= 2:
        health_status["status"] = "unhealthy"
    elif unhealthy_components or health_status["status"] == "degraded":
        health_status["status"] = "degraded"
    
    logger.info(f"🏥 Health check completed in {response_time:.3f}s - Status: {health_status['status']}")
    
    return health_status

@router.get("/components", status_code=status.HTTP_200_OK)
async def component_status() -> Dict[str, Any]:
    """
    Get detailed status of individual system components.
    """
    components = {}
    
    # Test each component individually
    component_tests = [
        ("database", _test_database),
        ("embedding", _test_embedding_service),
        ("knowledge_base", _test_knowledge_base),
        ("dspy", _test_dspy_service),
        ("mcp", _test_mcp_service)
    ]
    
    for component_name, test_func in component_tests:
        try:
            result = await test_func()
            components[component_name] = result
        except Exception as e:
            components[component_name] = {
                "status": "error",
                "message": f"Test function failed: {str(e)}",
                "details": {}
            }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "components": components
    }

@router.get("/metrics", status_code=status.HTTP_200_OK)
async def system_metrics(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Get system performance and usage metrics.
    """
    try:
        from app.services.db_service import DBService
        
        svc = DBService(db)
        
        # Get recent activity
        history = await svc.list_history(limit=100)
        feedback = await svc.list_feedback(limit=100)
        
        # Calculate metrics
        if history:
            avg_response_time = sum(h.get("response_time", 0) for h in history) / len(history)
            route_distribution = {}
            for h in history:
                route = h.get("source", "unknown")
                route_distribution[route] = route_distribution.get(route, 0) + 1
        else:
            avg_response_time = 0
            route_distribution = {}
        
        if feedback:
            ratings = [f.rating for f in feedback if f.rating]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
        else:
            avg_rating = 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "usage": {
                "total_queries": len(history),
                "total_feedback": len(feedback),
                "avg_response_time": round(avg_response_time, 3),
                "avg_user_rating": round(avg_rating, 2)
            },
            "routing": {
                "distribution": route_distribution,
                "total_routes": len(route_distribution)
            },
            "system": {
                "embedding_dimension": settings.EMBEDDING_DIMENSION,
                "kb_threshold": settings.KB_THRESHOLD,
                "max_response_time": settings.MAX_RESPONSE_TIME
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Metrics collection failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "usage": {},
            "routing": {},
            "system": {}
        }

@router.get("/readiness", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes-style readiness probe.
    Checks if the service is ready to handle requests.
    """
    ready = True
    checks = {}
    
    # Essential services that must be working
    essential_checks = [
        ("config", _check_configuration),
        ("embedding", _test_embedding_basic)
    ]
    
    for check_name, check_func in essential_checks:
        try:
            result = await check_func()
            checks[check_name] = result
            if not result.get("ready", False):
                ready = False
        except Exception as e:
            checks[check_name] = {"ready": False, "error": str(e)}
            ready = False
    
    status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return {
        "ready": ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/liveness", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes-style liveness probe.
    Basic check to ensure the service is alive.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time()  # Simple uptime indicator
    }

# ===== HELPER FUNCTIONS =====

async def _test_database() -> Dict[str, Any]:
    """Test database connectivity."""
    try:
        from app.db import engine
        
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1 as test")
            row = result.fetchone()
            
        return {
            "status": "healthy",
            "message": "Database connection successful",
            "details": {"test_query": "passed"}
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
            "details": {"error": str(e)}
        }

async def _test_embedding_service() -> Dict[str, Any]:
    """Test embedding service functionality."""
    try:
        from app.embedding_client import get_embedding, get_cache_info
        
        # Test embedding generation
        test_embedding = get_embedding("test query")
        
        # Get cache info
        cache_info = get_cache_info()
        
        return {
            "status": "healthy",
            "message": f"Embedding service operational",
            "details": {
                "embedding_dimension": len(test_embedding),
                "cache_info": cache_info,
                "model": settings.EMBEDDING_MODEL
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Embedding service failed: {str(e)}",
            "details": {"error": str(e)}
        }

async def _test_knowledge_base() -> Dict[str, Any]:
    """Test knowledge base accessibility."""
    try:
        from app.lancedb_store import LanceDBStore
        
        store = LanceDBStore()
        
        if store.table_exists():
            table_info = store.get_table_info()
            return {
                "status": "healthy",
                "message": "Knowledge base accessible",
                "details": table_info
            }
        else:
            return {
                "status": "degraded",
                "message": "Knowledge base table not found",
                "details": {"table_exists": False}
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Knowledge base test failed: {str(e)}",
            "details": {"error": str(e)}
        }

async def _test_dspy_service() -> Dict[str, Any]:
    """Test DSPy service initialization."""
    try:
        from app.dspy_integration import MathCoTPipeline
        
        # Test initialization only
        pipeline = MathCoTPipeline(api_key=settings.GOOGLE_API_KEY)
        
        return {
            "status": "healthy",
            "message": "DSPy pipeline ready",
            "details": {
                "model": settings.DSPY_MODEL,
                "temperature": settings.DSPY_TEMPERATURE
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"DSPy service failed: {str(e)}",
            "details": {"error": str(e)}
        }

async def _test_mcp_service() -> Dict[str, Any]:
    """Test MCP service availability."""
    try:
        from app.mcp_client import get_cache_stats
        
        cache_stats = get_cache_stats()
        
        return {
            "status": "healthy",
            "message": "MCP service ready",
            "details": {
                "cache_stats": cache_stats,
                "api_configured": bool(settings.TAVILY_API_KEY)
            }
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "message": f"MCP service test failed: {str(e)}",
            "details": {"error": str(e)}
        }

async def _check_configuration() -> Dict[str, Any]:
    """Check system configuration validity."""
    try:
        validation = settings.validate_config()
        
        return {
            "ready": validation["valid"],
            "message": "Configuration validation complete",
            "details": validation
        }
        
    except Exception as e:
        return {
            "ready": False,
            "message": f"Configuration check failed: {str(e)}",
            "details": {"error": str(e)}
        }

async def _test_embedding_basic() -> Dict[str, Any]:
    """Basic embedding test for readiness."""
    try:
        from app.embedding_client import _get_model
        
        # Just check if model loads
        model = _get_model()
        
        return {
            "ready": True,
            "message": "Embedding model loaded",
            "details": {"model_loaded": True}
        }
        
    except Exception as e:
        return {
            "ready": False,
            "message": f"Embedding model not ready: {str(e)}",
            "details": {"error": str(e)}
        }