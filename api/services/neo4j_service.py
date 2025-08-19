"""
Neo4j integration service for storing and querying knowledge graphs
"""
import logging
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver
from contextlib import contextmanager

from config import settings
from models import Entity, Relationship

logger = logging.getLogger(__name__)


class Neo4jService:
    """Service for Neo4j database operations"""
    
    def __init__(self):
        self.driver: Optional[Driver] = None
        self._initialized = False
        
    def initialize(self, max_retries: int = 5, retry_delay: int = 5):
        """Initialize Neo4j connection with retry logic"""
        if self._initialized:
            return
            
        import time
        
        for attempt in range(max_retries):
            try:
                logger.info("Connecting to Neo4j at %s (attempt %d/%d)", 
                           settings.neo4j_uri, attempt + 1, max_retries)
                
                self.driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password)
                )
                
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                
                logger.info("Neo4j connection established successfully")
                self._initialized = True
                return
                
            except Exception as e:
                logger.warning("Neo4j connection attempt %d failed: %s", attempt + 1, str(e))
                if attempt < max_retries - 1:
                    logger.info("Retrying Neo4j connection in %d seconds...", retry_delay)
                    time.sleep(retry_delay)
                else:
                    logger.error("Failed to connect to Neo4j after %d attempts", max_retries)
                    # Don't raise exception - allow graceful degradation
                    return
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self._initialized = False
            
    @contextmanager
    def get_session(self):
        """Get a Neo4j session with proper error handling"""
        if not self._initialized:
            self.initialize()
            
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            if not self._initialized:
                self.initialize()
            
            with self.get_session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
                
        except Exception as e:
            logger.error("Neo4j connection test failed: %s", str(e))
            return False
    
    def generate_merge_query_with_metadata(
        self, 
        entities: List[str], 
        entity_metadata: Dict[str, Entity]
    ) -> str:
        """Generate Cypher MERGE query with UMLS metadata"""
        queries = []
        
        for i, entity_name in enumerate(entities):
            # Escape quotes in entity name
            escaped_name = entity_name.replace('"', '\\"')
            query = f'MERGE (e{i}:MedicalEntity {{name: "{escaped_name}"'
            
            # Add metadata if available
            if entity_name in entity_metadata:
                entity = entity_metadata[entity_name]
                if entity.cui:
                    query += f', cui: "{entity.cui}"'
                if entity.canonical_name:
                    escaped_canonical = entity.canonical_name.replace('"', '\\"')
                    query += f', canonical_name: "{escaped_canonical}"'
                if entity.semantic_types:
                    types_str = '|'.join(entity.semantic_types)
                    query += f', semantic_types: "{types_str}"'
                if entity.linking_score is not None:
                    query += f', linking_score: {entity.linking_score}'
                if entity.description:
                    escaped_desc = entity.description.replace('"', '\\"')[:200]  # Limit description length
                    query += f', description: "{escaped_desc}"'
            
            query += '})'
            queries.append(query)
        
        return '\n'.join(queries)
    
    def generate_merge_relationships(
        self, 
        relationships: List[Relationship], 
        merge_entity_queries: str
    ) -> str:
        """Generate Cypher MERGE statements for relationships"""
        # Parse entity name to variable mapping
        entity_var_map = {}
        for match in re.finditer(r'MERGE\s*\((e\d+):MedicalEntity\s*\{\s*name:\s*"((?:[^"\\]|\\.)*)"', merge_entity_queries):
            var, name = match.group(1), match.group(2).replace('\\"', '"')
            entity_var_map[name] = var
        
        def format_relation(relation: str) -> str:
            return relation.lower().replace(" ", "_").replace('"', '\\"')
        
        seen = set()
        cypher_lines = []
        
        for rel in relationships:
            var1 = entity_var_map.get(rel.entity1)
            var2 = entity_var_map.get(rel.entity2)
            
            if var1 and var2:
                formatted_relation = format_relation(rel.relation)
                key = (var1, formatted_relation, var2)
                
                if key not in seen:
                    cypher_lines.append(
                        f'MERGE ({var1})-[:RELATIONSHIP {{type: "{formatted_relation}"}}]->({var2})'
                    )
                    seen.add(key)
        
        return '\n'.join(cypher_lines)
    
    def store_graph(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship], 
        note_id: Optional[str] = None
    ) -> Tuple[int, int, str]:
        """Store entities and relationships in Neo4j"""
        if not note_id:
            note_id = str(uuid.uuid4())
        
        try:
            # Get unique entities from relationships
            unique_entities = set()
            for rel in relationships:
                unique_entities.add(rel.entity1)
                unique_entities.add(rel.entity2)
            
            unique_entities_list = list(unique_entities)
            
            # Create entity metadata map
            entity_metadata = {ent.text: ent for ent in entities}
            
            # Generate Cypher queries
            cypher_entity_query = self.generate_merge_query_with_metadata(
                unique_entities_list, entity_metadata
            )
            cypher_relationship_query = self.generate_merge_relationships(
                relationships, cypher_entity_query
            )
            
            nodes_created = 0
            relationships_created = 0
            
            with self.get_session() as session:
                # Create entities
                if cypher_entity_query:
                    logger.debug("Executing entity creation query")
                    result = session.run(cypher_entity_query)
                    summary = result.consume()
                    nodes_created = summary.counters.nodes_created
                
                # Create relationships
                if cypher_relationship_query:
                    logger.debug("Executing relationship creation query")
                    result = session.run(cypher_relationship_query)
                    summary = result.consume()
                    relationships_created = summary.counters.relationships_created
            
            logger.info(
                "Stored graph for note %s: %d nodes, %d relationships", 
                note_id, nodes_created, relationships_created
            )
            
            return nodes_created, relationships_created, note_id
            
        except Exception as e:
            logger.error("Failed to store graph: %s", str(e))
            raise
    
    def query_graph(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        if parameters is None:
            parameters = {}
            
        try:
            with self.get_session() as session:
                logger.debug("Executing query: %s", query)
                result = session.run(query, parameters)
                records = [record.data() for record in result]
                
            logger.info("Query returned %d records", len(records))
            return records
            
        except Exception as e:
            logger.error("Query execution failed: %s", str(e))
            raise
    
    def clear_graph(self) -> int:
        """Clear all nodes and relationships from the graph"""
        try:
            with self.get_session() as session:
                result = session.run("MATCH (n) DETACH DELETE n")
                summary = result.consume()
                nodes_deleted = summary.counters.nodes_deleted
                
            logger.info("Cleared graph: %d nodes deleted", nodes_deleted)
            return nodes_deleted
            
        except Exception as e:
            logger.error("Failed to clear graph: %s", str(e))
            raise
    
    def get_entity_by_cui(self, cui: str) -> Optional[Dict[str, Any]]:
        """Get entity information by UMLS CUI"""
        try:
            query = "MATCH (e:MedicalEntity {cui: $cui}) RETURN e"
            results = self.query_graph(query, {"cui": cui})
            return results[0]["e"] if results else None
            
        except Exception as e:
            logger.error("Failed to get entity by CUI %s: %s", cui, str(e))
            raise


# Global service instance
neo4j_service = Neo4jService()