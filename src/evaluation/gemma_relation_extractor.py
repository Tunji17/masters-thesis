import json
import re
from typing import List, Dict
from dataclasses import dataclass
from mlx_lm import load, generate
from biored_data_loader import Document, BioREDDataLoader


@dataclass
class RelationPrediction:
    """Represents a predicted relation."""
    entity1_text: str
    entity2_text: str
    relation_type: str
    confidence: float = 0.0


class GemmaRelationExtractor:
    """Base class for Gemma-based relation extraction."""

    def __init__(self, model_name: str, device: str = "mps"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the Gemma model and tokenizer."""
        print(f"Loading {self.model_name} mlx model...")
        try:

            mlx_model, mlx_tokenizer = load(self.model_name)

            self.tokenizer = mlx_tokenizer
            self.model = mlx_model

            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from the model."""
        messages = [
            {
                "role": "system",
                "content": """You are an expert clinical information extraction
                system specialized in identifying medical relationships."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            # Use mlx_lm for generation
            response = generate(
                self.model,
                self.tokenizer,
                inputs,
                verbose=False,
                max_tokens=max_length,
            )

            # Clean up the response
            response = response.strip()
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            return response

        except Exception as e:
            print(f"Error generating response: {e}")
            return ""


class PromptBasedGemmaExtractor(GemmaRelationExtractor):
    """Prompt-based relation extraction using Gemma models."""

    def __init__(self, model_name: str, prompt_strategy: str = "basic", device: str = "mps"):
        super().__init__(model_name, device)
        self.prompt_strategy = prompt_strategy

    def create_prompt(self, document: Document, strategy: str = "basic", has_entity_context: bool = True) -> str:
        """Create prompts for relation extraction."""

        # Create entity context
        entity_context = "Entities mentioned in the text:\n"
        for entity in document.entities:
            entity_context += f"- {entity.text} ({entity.entity_type})\n"

        text = f"Title: {document.title}\n\nAbstract: {document.abstract}"

        entities = f"""You are provided with a list of medical
        entities extracted from the note:\n {entity_context}""" if has_entity_context else ""

        if strategy == "basic":
            prompt = f"""Your goal is to perform a Closed Information Extraction task on the following clinical note:

{text}

{entities}

RELATION TYPES:
- Association: General association between entities
- Positive_Correlation: Entity1 increases/enhances Entity2
- Negative_Correlation: Entity1 decreases/inhibits Entity2
- Bind: Physical binding between entities
- Cotreatment: Entities used together in treatment
- Comparison: Comparing effectiveness of entities
- Drug_Interaction: Interaction between drugs
- Conversion: One entity converts to another

Your task is to generate high quality triplets of the form (entity1, relation, entity2) where:
- The relationship is explicitly stated or strongly implied in the clinical note
- The entities are from the provided list (use the exact text as it appears)
- The triplets should be clinically meaningful and relevant

Please return the triplets in the following JSON format:
[
  {{
    "entity1": "entity name",
    "entity2": "entity name",
    "relation": "relation_type"
  }}
  ...
]"""

        elif strategy == "few_shot":
            prompt = f"""Your goal is to perform a Closed Information Extraction task on the following clinical note:

{text}

{entities}

RELATION TYPES:
- Association: General association between entities
- Positive_Correlation: Entity1 increases/enhances Entity2
- Negative_Correlation: Entity1 decreases/inhibits Entity2
- Bind: Physical binding between entities
- Cotreatment: Entities used together in treatment
- Comparison: Comparing effectiveness of entities
- Drug_Interaction: Interaction between drugs
- Conversion: One entity converts to another

Your task is to generate high quality triplets of the form (entity1, relation, entity2) where:
- The relationship is explicitly stated or strongly implied in the clinical note
- The entities are from the provided list (use the exact text as it appears)
- The triplets should be clinically meaningful and relevant


Some few-shot examples to guide you:
Example 1:
Text: "Aspirin treatment reduced inflammation in patients with arthritis."
Entities: aspirin (ChemicalEntity), inflammation (DiseaseOrPhenotypicFeature), arthritis (DiseaseOrPhenotypicFeature)
Relations: [{{"entity1": "aspirin", "entity2": "inflammation", "relation": "Negative_Correlation"}}]

Example 2:
Text: "The protein binds to DNA and regulates gene expression."
Entities: protein (GeneOrGeneProduct), DNA (GeneOrGeneProduct), gene expression (GeneOrGeneProduct)
Relations: [{{"entity1": "protein", "entity2": "DNA", "relation": "Bind"}}]

Please return the triplets in the following JSON format:
[
  {{
    "entity1": "entity name",
    "entity2": "entity name",
    "relation": "relation_type"
  }}
  ...
]"""

        elif strategy == "structured":
            prompt = f"""BIOMEDICAL RELATION EXTRACTION TASK

INPUT TEXT:
{text}

AVAILABLE ENTITIES:
{entities}

RELATION TYPES:
- Association: General association between entities
- Positive_Correlation: Entity1 increases/enhances Entity2
- Negative_Correlation: Entity1 decreases/inhibits Entity2
- Bind: Physical binding between entities
- Cotreatment: Entities used together in treatment
- Comparison: Comparing effectiveness of entities
- Drug_Interaction: Interaction between drugs
- Conversion: One entity converts to another

INSTRUCTIONS:
1. Identify pairs of entities that have relationships
2. Determine the most appropriate relation type
3. Only extract relations explicitly stated or strongly implied

OUTPUT FORMAT (JSON):
[
  {{
    "entity1": "exact entity text",
    "entity2": "exact entity text",
    "relation": "relation_type"
  }}
]"""

        if has_entity_context:
            print(f"Entities in document: {entity_context}")
        return prompt

    def dedupe(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove exact duplicates while preserving order."""
        seen = set()
        result = []
        for item in items:
            tup = (item["entity1"], item["entity2"], item["relation"])
            if tup not in seen:
                seen.add(tup)
                result.append(item)
        return result

    def parse_response(self, text: str) -> List[Dict[str, str]]:
        TRIPLE_KEYS = {"entity1", "entity2", "relation"}
        try:
            data = json.loads(text)
            if isinstance(data, list):
                triples = [
                    {
                        "entity1": str(item["entity1"]).strip(),
                        "entity2": str(item["entity2"]).strip(),
                        "relation": str(item["relation"]).strip(),
                    }
                    for item in data
                    if isinstance(item, dict) and TRIPLE_KEYS <= item.keys()
                ]
                if triples:
                    relations = self.dedupe(triples)
                    return relations
        except Exception:
            pass

        pattern = re.compile(
            r'"?entity1"?\s*:\s*"(?P<entity1>[^"]+)"[^{}]*?'
            r'"?entity2"?\s*:\s*"(?P<entity2>[^"]+)"[^{}]*?'
            r'"?relation"?\s*:\s*"(?P<relation>[^"]+)"',
            re.IGNORECASE | re.DOTALL,
        )

        triples = [
            {
                "entity1": m.group("entity1").strip(),
                "entity2": m.group("entity2").strip(),
                "relation": m.group("relation").strip(),
            }
            for m in pattern.finditer(text)
        ]
        relations = self.dedupe(triples)
        return relations

    def extract_relations_from_document(self, document: Document,
                                        has_entity_context: bool = True) -> List[RelationPrediction]:
        """Extract relations from a document using prompt-based approach."""
        prompt = self.create_prompt(document, self.prompt_strategy, has_entity_context)
        # print(f"Generated prompt:\n{prompt}\n")
        response = self.generate_response(prompt)

        print(f"Model response:\n{response}\n")
        return self.parse_response(response)


class GemmaEvaluator:
    """Evaluator for comparing Gemma and MedGemma models."""

    def __init__(self, data_loader: BioREDDataLoader):
        self.data_loader = data_loader
        self.models = {}

    def add_model(self, name: str, model_path: str, prompt_strategy: str = "basic"):
        """Add a model to the evaluation."""
        self.models[name] = {
            'path': model_path,
            'strategy': prompt_strategy,
            'extractor': None
        }

    def initialize_model(self, name: str):
        """Initialize a specific model."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")

        model_info = self.models[name]
        model_info['extractor'] = PromptBasedGemmaExtractor(
            model_name=model_info['path'],
            prompt_strategy=model_info['strategy']
        )
