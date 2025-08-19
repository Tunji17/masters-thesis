"""
API routes for entity and relationship extraction
"""
import logging
import uuid
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from models import (
    EntityRequest, EntityResponse, RelationshipRequest, RelationshipResponse,
    FullExtractionRequest, FullExtractionResponse, ErrorResponse
)
from services.entity_extractor import entity_service
from services.relation_extractor import relation_service
from services.neo4j_service import neo4j_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/extract", tags=["extraction"])


@router.post("/entities", response_model=EntityResponse)
async def extract_entities(request: EntityRequest):
    """Extract medical entities from clinical text"""
    try:
        entities = entity_service.extract_entities(
            text=request.text,
            threshold=request.threshold
        )
        
        return EntityResponse(
            entities=entities,
            count=len(entities)
        )
        
    except Exception as e:
        logger.error("Entity extraction failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Entity extraction failed: {str(e)}"
        )


@router.post("/relationships", response_model=RelationshipResponse)
async def extract_relationships(request: RelationshipRequest):
    """Extract relationships between medical entities"""
    try:
        relationships = relation_service.extract_relationships(
            text=request.text,
            entities=request.entities,
            max_tokens=request.max_tokens
        )
        
        return RelationshipResponse(
            relationships=relationships,
            count=len(relationships)
        )
        
    except Exception as e:
        logger.error("Relationship extraction failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Relationship extraction failed: {str(e)}"
        )


@router.post("/full", response_model=FullExtractionResponse)
async def full_extraction(request: FullExtractionRequest):
    """Complete extraction pipeline: entities + relationships + optional Neo4j storage"""
    note_id = str(uuid.uuid4())
    
    try:
        # Step 1: Extract entities
        logger.info("Starting full extraction for note %s", note_id)
        entities = entity_service.extract_entities(
            text=request.text,
            threshold=request.threshold
        )
        
        if not entities:
            logger.warning("No entities found for note %s", note_id)
            return FullExtractionResponse(
                note_id=note_id,
                entities=[],
                relationships=[],
                entities_count=0,
                relationships_count=0,
                stored=False
            )
        
        # Step 2: Extract relationships
        relationships = relation_service.extract_relationships(
            text=request.text,
            entities=entities,
            max_tokens=request.max_tokens
        )
        
        # Step 3: Store in Neo4j if requested
        cypher_query = None
        stored = False
        nodes_created = 0
        relationships_created = 0
        
        if request.store_in_graph and relationships:
            try:
                # Get unique entities from relationships
                unique_entities = set()
                for rel in relationships:
                    unique_entities.add(rel.entity1)
                    unique_entities.add(rel.entity2)
                
                # Create metadata map
                entity_metadata = {ent.text: ent for ent in entities}
                
                # Generate Cypher query for reference
                cypher_query = neo4j_service.generate_merge_query_with_metadata(
                    list(unique_entities), entity_metadata
                )
                cypher_relationships = neo4j_service.generate_merge_relationships(
                    relationships, cypher_query
                )
                cypher_query = cypher_query + "\n" + cypher_relationships
                
                # Store in Neo4j
                nodes_created, relationships_created, _ = neo4j_service.store_graph(
                    entities=entities,
                    relationships=relationships,
                    note_id=note_id
                )
                stored = True
                logger.info(
                    "Stored graph for note %s: %d nodes, %d relationships",
                    note_id, nodes_created, relationships_created
                )
                
            except Exception as e:
                logger.error("Failed to store in Neo4j: %s", str(e))
                # Continue without storing, don't fail the entire request
        
        return FullExtractionResponse(
            note_id=note_id,
            entities=entities,
            relationships=relationships,
            entities_count=len(entities),
            relationships_count=len(relationships),
            cypher_query=cypher_query,
            stored=stored
        )
        
    except Exception as e:
        logger.error("Full extraction failed for note %s: %s", note_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Full extraction failed: {str(e)}"
        )


@router.post("/batch/entities")
async def batch_extract_entities(requests: List[EntityRequest]):
    """Batch entity extraction for multiple texts"""
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 10 requests"
        )
    
    results = []
    
    for i, request in enumerate(requests):
        try:
            entities = entity_service.extract_entities(
                text=request.text,
                threshold=request.threshold
            )
            results.append({
                "index": i,
                "success": True,
                "entities": entities,
                "count": len(entities)
            })
            
        except Exception as e:
            logger.error("Batch entity extraction failed for request %d: %s", i, str(e))
            results.append({
                "index": i,
                "success": False,
                "error": str(e),
                "entities": [],
                "count": 0
            })
    
    return {"results": results}


@router.get("/cache/info")
async def get_cache_info():
    """Get cache statistics"""
    try:
        cache_info = entity_service.get_cache_info()
        return {
            "entity_cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "current_size": cache_info.currsize,
                "max_size": cache_info.maxsize
            }
        }
    except Exception as e:
        logger.error("Failed to get cache info: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get cache information"
        )


@router.post("/cache/clear")
async def clear_cache():
    """Clear the entity extraction cache"""
    try:
        entity_service.clear_cache()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error("Failed to clear cache: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to clear cache"
        )