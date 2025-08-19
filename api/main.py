"""
Medical Graph Extraction API

FastAPI application for extracting medical entities and relationships from clinical notes
with knowledge graph storage in Neo4j.
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from __init__ import __version__
from config import settings
from models import ErrorResponse
from routers import extraction, graph, health
from services.entity_extractor import entity_service
from services.relation_extractor import relation_service
from services.neo4j_service import neo4j_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    
    # Startup
    logger.info("Starting Medical Graph Extraction API v%s", __version__)
    
    # Initialize services
    try:
        logger.info("Initializing models and services...")
        
        # Initialize models in background to avoid blocking startup
        async def init_models():
            try:
                logger.info("Loading entity extraction models...")
                entity_service.initialize()
                logger.info("Entity extraction models loaded successfully")
                
                logger.info("Loading relation extraction model...")
                relation_service.initialize()
                logger.info("Relation extraction model loaded successfully")
                
            except Exception as e:
                logger.error("Failed to initialize models: %s", str(e))
                # Don't fail startup, allow graceful degradation
        
        # Initialize Neo4j connection
        try:
            logger.info("Connecting to Neo4j...")
            neo4j_service.initialize()
            logger.info("Neo4j connection established")
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            # Continue without Neo4j, allow graceful degradation
        
        # Start model loading in background
        asyncio.create_task(init_models())
        
        logger.info("API startup completed")
        
    except Exception as e:
        logger.error("Critical startup failure: %s", str(e))
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Medical Graph Extraction API")
    try:
        neo4j_service.close()
        logger.info("Neo4j connection closed")
    except Exception as e:
        logger.error("Error during shutdown: %s", str(e))


# Create FastAPI application
app = FastAPI(
    title="Medical Graph Extraction API",
    description="""
    A FastAPI service for extracting medical entities and relationships from clinical notes.
    
    ## Features
    
    * **Entity Extraction**: Extract medical entities using GLiNER with UMLS linking via SciSpacy
    * **Relationship Extraction**: Generate medical relationships using Gemma LLM
    * **Knowledge Graph Storage**: Store extracted knowledge in Neo4j graph database
    * **Batch Processing**: Process multiple clinical notes efficiently
    * **Comprehensive API**: Full CRUD operations for graph data
    
    ## Workflow
    
    1. **Entity Extraction**: Clinical text → GLiNER → SciSpacy UMLS linking → Enhanced entities
    2. **Relationship Extraction**: Entities + Clinical text → Gemma → Medical relationships  
    3. **Graph Storage**: Entities + Relationships → Neo4j → Knowledge graph
    
    ## Models Used
    
    * **MLX NER**: Pattern-based medical entity extraction (replaces GLiNER)
    * **SciSpacy**: en_core_sci_lg (UMLS entity linking)
    * **Gemma**: google/gemma-3-4b-it with MLX (Relationship extraction)
    """,
    version=__version__,
    lifespan=lifespan,
    contact={
        "name": "Medical NLP Research",
        "email": "contact@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(extraction.router, prefix="/api")
app.include_router(graph.router, prefix="/api")


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.warning("Request validation error: %s", exc.errors())
    
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc.errors()),
            type="validation_error"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error("Unhandled exception: %s", str(exc), exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Please try again later.",
            type="internal_error"
        ).dict()
    )


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "Medical Graph Extraction API",
        "version": __version__,
        "description": "API for extracting medical entities and relationships from clinical notes",
        "docs": "/docs",
        "health": "/health",
        "status": "running"
    }


# Additional endpoints
@app.get("/api", tags=["root"])
async def api_info():
    """API information endpoint"""
    return {
        "api_version": __version__,
        "endpoints": {
            "entity_extraction": "/api/extract/entities",
            "relationship_extraction": "/api/extract/relationships", 
            "full_pipeline": "/api/extract/full",
            "graph_storage": "/api/graph/store",
            "graph_query": "/api/graph/query",
            "health_check": "/health"
        },
        "models": {
            "ner_model": "mlx-ner-medical",
            "scispacy": "en_core_sci_lg", 
            "gemma": "google/gemma-3-4b-it (MLX)"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False  # Set to True for development
    )