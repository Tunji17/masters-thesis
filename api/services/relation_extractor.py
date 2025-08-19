"""
Relationship extraction service using Gemma model with MLX
"""
import logging
import re
from typing import List, Tuple
from mlx_lm import load, generate

from config import settings
from models import Entity, Relationship

logger = logging.getLogger(__name__)


class RelationExtractionService:
    """Service for medical relationship extraction using Gemma model with MLX"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def initialize(self):
        """Initialize the Gemma model"""
        if self._initialized:
            return
            
        try:
            logger.info("Loading Gemma model with MLX: %s", settings.gemma_model)
            
            # Load model and tokenizer using MLX
            self.model, self.tokenizer = load(settings.gemma_model)
            
            logger.info("Gemma model loaded successfully with MLX")
            self._initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize Gemma model: %s", str(e))
            raise
    
    def _extract_triples(self, raw_text: str) -> List[Tuple[str, str, str]]:
        """Extract well-formed triples from model output"""
        tuple_pattern = re.compile(r'''\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)''')
        return [match for match in tuple_pattern.findall(raw_text)]
    
    def extract_relationships(
        self, 
        text: str, 
        entities: List[Entity], 
        max_tokens: int = 512
    ) -> List[Relationship]:
        """Extract relationships from clinical text and entities"""
        if not self._initialized:
            self.initialize()
            
        if not entities:
            logger.warning("No entities provided for relationship extraction")
            return []
        
        try:
            # Format entities with their medical knowledge
            entity_descriptions = []
            for ent in entities:
                desc = f'- "{ent.text}"'
                if ent.canonical_name:
                    desc += f" [UMLS: {ent.canonical_name}]"
                if ent.description:
                    desc += f" - {ent.description[:100]}..."
                entity_descriptions.append(desc)
            
            entities_text = "\n".join(entity_descriptions)
            
            # Create relationship extraction prompt
            relationship_prompt = f"""Your goal is to perform a Closed Information Extraction task on the following clinical note:

{text}

You are provided with a list of medical entities extracted from the note:
{entities_text}

Your task is to generate high quality triplets of the form (entity1, relation, entity2) where:
- The relationship is explicitly stated or strongly implied in the clinical note
- The entities are from the provided list (use the exact text as it appears)
- The triplets should be clinically meaningful and relevant

Please return the triplets in the following format:
[
  ("entity1", "relation", "entity2"),
  ("entity3", "relation", "entity4"),
  ...
]
"""
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert clinical information extraction system specialized in identifying medical relationships."
                },
                {
                    "role": "user",
                    "content": relationship_prompt
                }
            ]
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # Generate relationships with MLX
            logger.debug("Generating relationships with Gemma model using MLX")
            triplet_str = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            
            # Extract triples from generated text
            triples_list = self._extract_triples(triplet_str)
            
            # Convert to Relationship objects
            relationships = []
            for entity1, relation, entity2 in triples_list:
                relationships.append(Relationship(
                    entity1=entity1,
                    relation=relation,
                    entity2=entity2
                ))
            
            logger.info("Extracted %d relationships", len(relationships))
            return relationships
            
        except Exception as e:
            logger.error("Relationship extraction failed: %s", str(e))
            raise


# Global service instance
relation_service = RelationExtractionService()