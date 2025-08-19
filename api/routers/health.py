"""
Health check and monitoring endpoints
"""
import logging
from fastapi import APIRouter

from __init__ import __version__
from models import HealthResponse
from services.entity_extractor import entity_service
from services.relation_extractor import relation_service
from services.neo4j_service import neo4j_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for all services"""
    
    # Check model loading status
    models_status = {
        "gliner": entity_service._initialized,
        "scispacy": entity_service.nlp is not None if entity_service._initialized else False,
        "gemma": relation_service._initialized
    }
    
    # Check Neo4j connection
    neo4j_connected = neo4j_service.test_connection()
    
    # Determine overall status
    all_models_loaded = all(models_status.values())
    overall_status = "healthy" if all_models_loaded and neo4j_connected else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=__version__,
        models_loaded=models_status,
        neo4j_connected=neo4j_connected
    )


@router.get("/health/models")
async def models_health():
    """Detailed model status check"""
    
    models_info = {}
    
    # GLiNER model status
    try:
        if entity_service._initialized:
            models_info["gliner"] = {
                "status": "loaded",
                "model_name": "Ihor/gliner-biomed-bi-large-v1.0",
                "initialized": True
            }
        else:
            models_info["gliner"] = {
                "status": "not_loaded",
                "model_name": "Ihor/gliner-biomed-bi-large-v1.0", 
                "initialized": False
            }
    except Exception as e:
        models_info["gliner"] = {
            "status": "error",
            "error": str(e),
            "initialized": False
        }
    
    # SciSpacy model status
    try:
        if entity_service._initialized and entity_service.nlp:
            models_info["scispacy"] = {
                "status": "loaded",
                "model_name": "en_core_sci_lg",
                "pipeline_components": entity_service.nlp.pipe_names,
                "initialized": True
            }
        else:
            models_info["scispacy"] = {
                "status": "not_loaded",
                "model_name": "en_core_sci_lg",
                "initialized": False
            }
    except Exception as e:
        models_info["scispacy"] = {
            "status": "error",
            "error": str(e),
            "initialized": False
        }
    
    # Gemma model status
    try:
        if relation_service._initialized:
            models_info["gemma"] = {
                "status": "loaded",
                "model_name": "google/gemma-3-4b-it",
                "initialized": True
            }
        else:
            models_info["gemma"] = {
                "status": "not_loaded",
                "model_name": "google/gemma-3-4b-it",
                "initialized": False
            }
    except Exception as e:
        models_info["gemma"] = {
            "status": "error",
            "error": str(e),
            "initialized": False
        }
    
    return {"models": models_info}


@router.get("/health/neo4j")
async def neo4j_health():
    """Neo4j connection health check"""
    
    try:
        connected = neo4j_service.test_connection()
        
        neo4j_info = {
            "status": "connected" if connected else "disconnected",
            "uri": neo4j_service.driver.encrypted if neo4j_service.driver else "N/A",
            "initialized": neo4j_service._initialized
        }
        
        if connected:
            # Get basic database info
            try:
                db_info = neo4j_service.query_graph("CALL dbms.components()")
                if db_info:
                    neo4j_info["version"] = db_info[0].get("versions", ["Unknown"])[0]
                    neo4j_info["edition"] = db_info[0].get("edition", "Unknown")
            except:
                pass  # Don't fail health check if we can't get version info
        
        return {"neo4j": neo4j_info}
        
    except Exception as e:
        return {
            "neo4j": {
                "status": "error",
                "error": str(e),
                "initialized": neo4j_service._initialized
            }
        }


@router.get("/health/cache")
async def cache_health():
    """Cache statistics and health"""
    
    try:
        cache_info = entity_service.get_cache_info()
        
        return {
            "cache": {
                "status": "active",
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
                "current_size": cache_info.currsize,
                "max_size": cache_info.maxsize
            }
        }
        
    except Exception as e:
        return {
            "cache": {
                "status": "error",
                "error": str(e)
            }
        }


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes-style readiness probe"""
    
    # Check if all critical services are ready
    models_ready = (
        entity_service._initialized and
        entity_service.nlp is not None and
        relation_service._initialized
    )
    
    neo4j_ready = neo4j_service.test_connection()
    
    if models_ready and neo4j_ready:
        return {"status": "ready"}
    else:
        return {"status": "not_ready", "details": {
            "models_ready": models_ready,
            "neo4j_ready": neo4j_ready
        }}, 503  # Service Unavailable


@router.get("/health/live")
async def liveness_check():
    """Kubernetes-style liveness probe"""
    
    # Basic liveness check - service is running
    return {"status": "alive"}