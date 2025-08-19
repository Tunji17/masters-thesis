"""
Entity extraction service using MLX-based NER and SciSpacy
"""
import logging
import re
from typing import List, Dict, Any, Optional
from functools import lru_cache
import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector

from config import settings
from models import Entity

logger = logging.getLogger(__name__)


class MLXNERModel:
    """MLX-based NER model for medical entity extraction"""
    
    def __init__(self, model_name="mlx-ner-medical"):
        self.model_name = model_name
        # Enhanced rule-based patterns for medical entities
        self.entity_patterns = {
            "Disease or Condition": [
                r'\b(?:cancer|tumor|disease|syndrome|disorder|condition|infection|inflammation)\b',
                r'\b(?:diabetes|hypertension|pneumonia|asthma|bronchitis|carcinoma)\b',
                r'\b(?:coarctation|stenosis|insufficiency|regurgitation)\b',
                r'\b(?:acute|chronic|severe|mild|moderate)\s+(?:\w+\s+)?(?:disease|condition|syndrome)\b'
            ],
            "Medication": [
                r'\b(?:mg|mcg|tablet|capsule|injection|dose|dosage|medication)\b',
                r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|insulin|lisinopril)\b',
                r'\d+\s*(?:mg|mcg|ml|units?)\b'
            ],
            "Medication Dosage and Frequency": [
                r'\d+\s*(?:mg|mcg|ml|units?)(?:\s+(?:daily|twice|once|bid|tid|qid))?\b',
                r'\b(?:daily|twice daily|once daily|bid|tid|qid)\b'
            ],
            "Procedure": [
                r'\b(?:surgery|operation|procedure|biopsy|scan|test|examination)\b',
                r'\b(?:CT|MRI|X-ray|ultrasound|endoscopy|echocardiography|catheterization)\b',
                r'\bechocardiographic\s+study\b'
            ],
            "Lab Test": [
                r'\b(?:blood test|lab test|laboratory|CBC|BUN|creatinine|glucose)\b',
                r'\b(?:hemoglobin|hematocrit|platelet|WBC|RBC)\b'
            ],
            "Lab Test Result": [
                r'\d+(?:\.\d+)?\s*(?:mg/dl|mmol/L|%|units)\b',
                r'\b(?:normal|abnormal|elevated|decreased|low|high)\s+(?:levels?|values?)\b'
            ],
            "Body Site": [
                r'\b(?:heart|lung|liver|kidney|brain|chest|abdomen|aortic|cardiac)\b',
                r'\b(?:arm|leg|head|neck|back|extremities|ankle|upper extremities)\b',
                r'\b(?:right|left)\s+(?:arm|leg|ankle|extremit(?:y|ies))\b'
            ],
            "Medical Device": [
                r'\b(?:stent|catheter|pacemaker|defibrillator|prosthesis)\b',
                r'\b(?:monitor|ventilator|dialysis)\b'
            ],
            "Demographic Information": [
                r'\b(?:age|years old|male|female|gender|race|ethnicity)\b',
                r'\d+\s*(?:year|yr)s?\s+old\b'
            ]
        }
        
    def predict_entities(self, text: str, labels: List[str], threshold: float = 0.5) -> List[Dict]:
        """Predict entities using enhanced pattern matching"""
        entities = []
        
        for label in labels:
            if label in self.entity_patterns:
                for pattern in self.entity_patterns[label]:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities.append({
                            'text': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'label': label,
                            'score': 0.85  # Higher confidence score
                        })
        
        # Remove duplicates and overlapping entities
        unique_entities = []
        seen_spans = set()
        
        # Sort by start position and score (descending)
        for entity in sorted(entities, key=lambda x: (x['start'], -x['score'])):
            span = (entity['start'], entity['end'])
            # Check for overlap with existing entities
            overlap = any(
                (span[0] < existing[1] and span[1] > existing[0])
                for existing in seen_spans
            )
            if not overlap:
                seen_spans.add(span)
                unique_entities.append(entity)
        
        return unique_entities


class EntityExtractionService:
    """Service for medical entity extraction and linking"""
    
    def __init__(self):
        self.ner_model = None
        self.nlp = None
        self._initialized = False
        
    def initialize(self):
        """Initialize the models"""
        if self._initialized:
            return
            
        try:
            logger.info("Loading MLX-based NER model...")
            self.ner_model = MLXNERModel()
            logger.info("MLX-based NER model loaded successfully")
            
            logger.info("Loading SciSpacy model...")
            self.nlp = spacy.load(settings.scispacy_model)
            
            # Add abbreviation detector
            self.nlp.add_pipe("abbreviation_detector")
            
            # Add UMLS entity linker
            self.nlp.add_pipe("scispacy_linker", config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "max_entities_per_mention": 3
            })
            
            logger.info("SciSpacy pipeline loaded with components: %s", self.nlp.pipe_names)
            self._initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize entity extraction models: %s", str(e))
            raise
    
    @lru_cache(maxsize=settings.cache_size)
    def _cached_entity_lookup(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """Cache entity lookups to improve performance"""
        try:
            doc = self.nlp(entity_text)
            if doc.ents and doc.ents[0]._.kb_ents:
                ent = doc.ents[0]
                cui, score = ent._.kb_ents[0]
                linker = self.nlp.get_pipe("scispacy_linker")
                kb_entity = linker.kb.cui_to_entity[cui]
                return {
                    'cui': cui,
                    'score': score,
                    'name': kb_entity.canonical_name,
                    'definition': kb_entity.definition if kb_entity.definition else ''
                }
        except Exception as e:
            logger.warning("Entity lookup failed for '%s': %s", entity_text, str(e))
        return None
    
    def extract_entities(self, text: str, threshold: float = 0.5) -> List[Entity]:
        """Extract and enhance entities from clinical text"""
        if not self._initialized:
            self.initialize()
            
        # Labels for GLiNER model
        labels = [
            "Disease or Condition", "Medication", "Medication Dosage and Frequency",
            "Procedure", "Lab Test", "Lab Test Result", "Body Site",
            "Medical Device", "Demographic Information"
        ]
        
        try:
            # Extract entities with MLX-based NER
            logger.debug("Extracting entities with MLX-based NER")
            ner_entities = self.ner_model.predict_entities(
                text, labels=labels, threshold=threshold
            )
            
            # Process text with SciSpacy for entity linking
            logger.debug("Processing text with SciSpacy")
            doc = self.nlp(text)
            
            # Create mapping of entity text to SciSpacy linked entities
            entity_links = {}
            for ent in doc.ents:
                if ent._.kb_ents:
                    candidates = []
                    for umls_ent in ent._.kb_ents[:3]:  # Top 3 candidates
                        cui, score = umls_ent
                        linker = self.nlp.get_pipe("scispacy_linker")
                        kb_entity = linker.kb.cui_to_entity[cui]
                        candidates.append({
                            'cui': cui,
                            'score': score,
                            'name': kb_entity.canonical_name,
                            'definition': kb_entity.definition if kb_entity.definition else '',
                            'types': list(kb_entity.types)
                        })
                    entity_links[ent.text.lower()] = candidates
            
            # Enhance NER entities with SciSpacy linking
            enhanced_entities = []
            for entity in ner_entities:
                entity_text_lower = entity['text'].lower()
                
                enhanced_entity = Entity(
                    text=entity['text'],
                    start=entity['start'],
                    end=entity['end']
                )
                
                # Check if we have linking information
                if entity_text_lower in entity_links:
                    candidates = entity_links[entity_text_lower]
                    if candidates:
                        # Use the top candidate
                        top_candidate = candidates[0]
                        enhanced_entity.cui = top_candidate['cui']
                        enhanced_entity.canonical_name = top_candidate['name']
                        enhanced_entity.description = top_candidate['definition']
                        enhanced_entity.semantic_types = top_candidate['types']
                        enhanced_entity.linking_score = top_candidate['score']
                        enhanced_entity.alternative_candidates = candidates[1:] if len(candidates) > 1 else []
                
                enhanced_entities.append(enhanced_entity)
            
            # Check for abbreviations and their expansions
            abbreviations = {}
            for abbr in doc._.abbreviations:
                abbreviations[abbr.text] = abbr._.long_form.text
            
            # Add abbreviation information
            for entity in enhanced_entities:
                if entity.text in abbreviations:
                    entity.expanded_form = abbreviations[entity.text]
            
            # Remove duplicates based on text, keeping highest linking score
            unique_entities = {}
            for entity in enhanced_entities:
                key = entity.text
                if key not in unique_entities or \
                   (entity.linking_score and 
                    entity.linking_score > unique_entities[key].linking_score):
                    unique_entities[key] = entity
            
            result = list(unique_entities.values())
            logger.info("Extracted %d unique entities", len(result))
            return result
            
        except Exception as e:
            logger.error("Entity extraction failed: %s", str(e))
            raise
    
    def get_cache_info(self):
        """Get cache statistics"""
        return self._cached_entity_lookup.cache_info()
    
    def clear_cache(self):
        """Clear the entity cache"""
        self._cached_entity_lookup.cache_clear()


# Global service instance
entity_service = EntityExtractionService()