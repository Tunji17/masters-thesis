"""
API routes for Neo4j graph operations
"""
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from models import (
    GraphStoreRequest, GraphStoreResponse, GraphQueryRequest, GraphQueryResponse
)
from services.neo4j_service import neo4j_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("/store", response_model=GraphStoreResponse)
async def store_graph(request: GraphStoreRequest):
    """Store entities and relationships in Neo4j"""
    try:
        nodes_created, relationships_created, note_id = neo4j_service.store_graph(
            entities=request.entities,
            relationships=request.relationships,
            note_id=request.note_id
        )
        
        return GraphStoreResponse(
            success=True,
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            note_id=note_id
        )
        
    except Exception as e:
        logger.error("Graph storage failed: %s", str(e))
        return GraphStoreResponse(
            success=False,
            nodes_created=0,
            relationships_created=0,
            note_id=request.note_id or "unknown"
        )


@router.post("/query", response_model=GraphQueryResponse)
async def query_graph(request: GraphQueryRequest):
    """Execute a Cypher query on the Neo4j graph"""
    try:
        # Basic query validation
        query_lower = request.query.lower().strip()
        
        # Prevent destructive operations
        destructive_keywords = ['delete', 'drop', 'create constraint', 'create index']
        if any(keyword in query_lower for keyword in destructive_keywords):
            raise HTTPException(
                status_code=400,
                detail="Destructive operations not allowed through query endpoint"
            )
        
        results = neo4j_service.query_graph(
            query=request.query,
            parameters=request.parameters
        )
        
        return GraphQueryResponse(
            results=results,
            count=len(results)
        )
        
    except Exception as e:
        logger.error("Graph query failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Graph query failed: {str(e)}"
        )


@router.get("/entities/cui/{cui}")
async def get_entity_by_cui(cui: str):
    """Get entity information by UMLS CUI"""
    try:
        entity = neo4j_service.get_entity_by_cui(cui)
        
        if not entity:
            raise HTTPException(
                status_code=404,
                detail=f"Entity with CUI '{cui}' not found"
            )
        
        return {"entity": entity}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get entity by CUI %s: %s", cui, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get entity: {str(e)}"
        )


@router.get("/stats")
async def get_graph_stats():
    """Get basic statistics about the knowledge graph"""
    try:
        stats_queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "entity_types": "MATCH (n:MedicalEntity) RETURN count(n) as count",
            "relationship_types": "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count"
        }
        
        stats = {}
        
        # Get basic counts
        for stat_name, query in list(stats_queries.items())[:3]:
            results = neo4j_service.query_graph(query)
            stats[stat_name] = results[0]["count"] if results else 0
        
        # Get relationship type distribution
        rel_types_result = neo4j_service.query_graph(stats_queries["relationship_types"])
        stats["relationship_type_distribution"] = [
            {"type": r["type"], "count": r["count"]} 
            for r in rel_types_result
        ]
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get graph stats: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph statistics: {str(e)}"
        )


@router.get("/entities/search")
async def search_entities(
    name: Optional[str] = Query(None, description="Search by entity name"),
    cui: Optional[str] = Query(None, description="Filter by UMLS CUI"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return")
):
    """Search for entities in the knowledge graph"""
    try:
        conditions = []
        parameters = {"limit": limit}
        
        if name:
            conditions.append("toLower(e.name) CONTAINS toLower($name)")
            parameters["name"] = name
        
        if cui:
            conditions.append("e.cui = $cui")
            parameters["cui"] = cui
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"MATCH (e:MedicalEntity) WHERE {where_clause} RETURN e LIMIT $limit"
        
        results = neo4j_service.query_graph(query, parameters)
        entities = [record["e"] for record in results]
        
        return {
            "entities": entities,
            "count": len(entities),
            "total_found": len(entities)  # Note: This is limited by the LIMIT clause
        }
        
    except Exception as e:
        logger.error("Entity search failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Entity search failed: {str(e)}"
        )


@router.delete("/clear")
async def clear_graph():
    """Clear all data from the Neo4j graph database"""
    try:
        nodes_deleted = neo4j_service.clear_graph()
        
        return {
            "message": "Graph cleared successfully",
            "nodes_deleted": nodes_deleted
        }
        
    except Exception as e:
        logger.error("Failed to clear graph: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear graph: {str(e)}"
        )


@router.get("/schema")
async def get_graph_schema():
    """Get the schema/structure of the Neo4j graph"""
    try:
        schema_queries = {
            "node_labels": "CALL db.labels()",
            "relationship_types": "CALL db.relationshipTypes()",
            "property_keys": "CALL db.propertyKeys()"
        }
        
        schema = {}
        for key, query in schema_queries.items():
            results = neo4j_service.query_graph(query)
            if key == "node_labels":
                schema[key] = [r["label"] for r in results]
            elif key == "relationship_types":
                schema[key] = [r["relationshipType"] for r in results]
            else:
                schema[key] = [r["propertyKey"] for r in results]
        
        return schema
        
    except Exception as e:
        logger.error("Failed to get graph schema: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph schema: {str(e)}"
        )