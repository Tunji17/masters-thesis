"""
Pydantic models for the Medical Graph Extraction API
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class EntityRequest(BaseModel):
    text: str = Field(..., description="Clinical text to extract entities from", min_length=1)
    threshold: Optional[float] = Field(0.5, description="Confidence threshold for entity extraction", ge=0.0, le=1.0)


class Entity(BaseModel):
    text: str = Field(..., description="Entity text as it appears in the source")
    start: int = Field(..., description="Starting character position", ge=0)
    end: int = Field(..., description="Ending character position", ge=0)
    cui: Optional[str] = Field(None, description="UMLS CUI identifier")
    canonical_name: Optional[str] = Field(None, description="UMLS canonical name")
    description: Optional[str] = Field(None, description="Entity description from UMLS")
    semantic_types: Optional[List[str]] = Field(None, description="UMLS semantic types")
    linking_score: Optional[float] = Field(None, description="Entity linking confidence score")
    expanded_form: Optional[str] = Field(None, description="Expanded form of abbreviation")
    alternative_candidates: Optional[List[Dict[str, Any]]] = Field(None, description="Alternative UMLS candidates")


class EntityResponse(BaseModel):
    entities: List[Entity] = Field(..., description="Extracted entities")
    count: int = Field(..., description="Number of entities extracted")


class RelationshipRequest(BaseModel):
    text: str = Field(..., description="Clinical text for relationship extraction", min_length=1)
    entities: List[Entity] = Field(..., description="Extracted entities to use for relationship extraction")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens for model generation", ge=50, le=2048)


class Relationship(BaseModel):
    entity1: str = Field(..., description="First entity in the relationship")
    relation: str = Field(..., description="Relationship type")
    entity2: str = Field(..., description="Second entity in the relationship")


class RelationshipResponse(BaseModel):
    relationships: List[Relationship] = Field(..., description="Extracted relationships")
    count: int = Field(..., description="Number of relationships extracted")


class FullExtractionRequest(BaseModel):
    text: str = Field(..., description="Clinical text to process", min_length=1)
    threshold: Optional[float] = Field(0.5, description="Entity extraction threshold", ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(512, description="Max tokens for relationship extraction", ge=50, le=2048)
    store_in_graph: Optional[bool] = Field(True, description="Whether to store results in Neo4j")


class FullExtractionResponse(BaseModel):
    note_id: str = Field(..., description="Generated note ID")
    entities: List[Entity] = Field(..., description="Extracted entities")
    relationships: List[Relationship] = Field(..., description="Extracted relationships")
    entities_count: int = Field(..., description="Number of entities")
    relationships_count: int = Field(..., description="Number of relationships")
    cypher_query: Optional[str] = Field(None, description="Generated Cypher query for Neo4j")
    stored: bool = Field(..., description="Whether data was stored in Neo4j")


class GraphStoreRequest(BaseModel):
    entities: List[Entity] = Field(..., description="Entities to store")
    relationships: List[Relationship] = Field(..., description="Relationships to store")
    note_id: Optional[str] = Field(None, description="Note identifier")


class GraphStoreResponse(BaseModel):
    success: bool = Field(..., description="Whether storage was successful")
    nodes_created: int = Field(..., description="Number of nodes created")
    relationships_created: int = Field(..., description="Number of relationships created")
    note_id: str = Field(..., description="Note identifier")


class GraphQueryRequest(BaseModel):
    query: str = Field(..., description="Cypher query to execute", min_length=1)
    parameters: Optional[Dict[str, Any]] = Field({}, description="Query parameters")


class GraphQueryResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    count: int = Field(..., description="Number of results returned")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    neo4j_connected: bool = Field(..., description="Neo4j connection status")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    type: str = Field("error", description="Error type")