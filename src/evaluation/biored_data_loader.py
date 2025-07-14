"""
BioRED Dataset Loader
Loads biomedical relation annotations from PubTator format files.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entity:
    """Represents an entity annotation."""
    pmid: str
    start: int
    end: int
    text: str
    entity_type: str  # GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity, etc.
    entity_id: str


@dataclass
class Relation:
    """Represents a biomedical relation."""
    pmid: str
    relation_type: str  # Association, Positive_Correlation, etc.
    entity1_id: str
    entity2_id: str
    novel: str  # Yes/No - whether this is a novel relation


@dataclass
class Document:
    """Represents a PubMed document with title, abstract, entities, and relations."""
    pmid: str
    title: str
    abstract: str
    entities: List[Entity]
    relations: List[Relation]


class BioREDDataLoader:
    """Loads BioRED data from PubTator format files."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_pubtator_file(self, file_path: str) -> List[Document]:
        """Load documents from a PubTator format file."""
        documents = []
        current_doc = None

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    if current_doc:
                        documents.append(current_doc)
                        current_doc = None
                    continue

                # Parse title line
                if '|t|' in line:
                    pmid, title = line.split('|t|', 1)
                    current_doc = Document(
                        pmid=pmid,
                        title=title,
                        abstract='',
                        entities=[],
                        relations=[]
                    )

                # Parse abstract line
                elif '|a|' in line:
                    if current_doc:
                        pmid, abstract = line.split('|a|', 1)
                        current_doc.abstract = abstract

                # Parse entity annotation or relation
                elif '\t' in line and not line.startswith('#'):
                    parts = line.split('\t')

                    if len(parts) >= 3:
                        pmid = parts[0]

                        # Handle relation annotations (Association, Positive_Correlation, etc.)
                        if len(parts) >= 5 and parts[1] in ['Association', 'Positive_Correlation',
                                                            'Negative_Correlation', 'Bind', 'Cotreatment',
                                                            'Comparison', 'Drug_Interaction', 'Conversion']:
                            relation_type = parts[1]
                            entity1_id = parts[2]
                            entity2_id = parts[3]
                            novel = parts[4] if len(parts) > 4 else 'No'

                            if current_doc:
                                relation = Relation(
                                    pmid=pmid,
                                    relation_type=relation_type,
                                    entity1_id=entity1_id,
                                    entity2_id=entity2_id,
                                    novel=novel
                                )
                                current_doc.relations.append(relation)

                        # Handle entity annotations
                        elif len(parts) >= 6:
                            try:
                                start = int(parts[1])
                                end = int(parts[2])
                                text = parts[3]
                                entity_type = parts[4]
                                entity_id = parts[5] if len(parts) > 5 else ''

                                if current_doc:
                                    entity = Entity(
                                        pmid=pmid,
                                        start=start,
                                        end=end,
                                        text=text,
                                        entity_type=entity_type,
                                        entity_id=entity_id
                                    )
                                    current_doc.entities.append(entity)
                            except (ValueError, IndexError):
                                # Skip malformed lines
                                continue

            # Add the last document if exists
            if current_doc:
                documents.append(current_doc)

        return documents

    def load_training_data(self) -> List[Document]:
        """Load training data."""
        train_file = self.data_path / 'Train.PubTator'
        return self.load_pubtator_file(str(train_file))

    def load_development_data(self) -> List[Document]:
        """Load development data."""
        dev_file = self.data_path / 'Dev.PubTator'
        return self.load_pubtator_file(str(dev_file))

    def load_test_data(self) -> List[Document]:
        """Load test data."""
        test_file = self.data_path / 'Test.PubTator'
        return self.load_pubtator_file(str(test_file))

    def get_dataset_statistics(self, documents: List[Document]) -> Dict[str, int]:
        """Get basic statistics about the dataset."""
        # Count entity types
        entity_type_counts = {}
        for doc in documents:
            for entity in doc.entities:
                entity_type_counts[entity.entity_type] = entity_type_counts.get(entity.entity_type, 0) + 1

        # Count relation types
        relation_type_counts = {}
        for doc in documents:
            for relation in doc.relations:
                relation_type_counts[relation.relation_type] = relation_type_counts.get(relation.relation_type, 0) + 1

        stats = {
            'total_documents': len(documents),
            'total_entities': sum(len(doc.entities) for doc in documents),
            'total_relations': sum(len(doc.relations) for doc in documents),
            'entity_types': entity_type_counts,
            'relation_types': relation_type_counts
        }
        return stats

    def extract_relation_triples(self, documents: List[Document]) -> List[Tuple[str, str, str, str]]:
        """Extract relation triples in format (pmid, entity1_text, entity2_text, relation_type)."""
        triples = []

        for doc in documents:
            # Create entity lookup by ID
            entity_lookup = {}
            for entity in doc.entities:
                # Handle multiple IDs separated by comma
                for entity_id in entity.entity_id.split(','):
                    entity_lookup[entity_id.strip()] = entity

            for relation in doc.relations:
                entity1 = entity_lookup.get(relation.entity1_id)
                entity2 = entity_lookup.get(relation.entity2_id)

                if entity1 and entity2:
                    triple = (
                        doc.pmid,
                        entity1.text,
                        entity2.text,
                        relation.relation_type
                    )
                    triples.append(triple)

        print(f"first 5 triples extracted: {triples[:5]}")
        return triples

    def filter_relations_by_type(self, documents: List[Document], relation_types: List[str]) -> List[Document]:
        """Filter documents to only include specified relation types."""
        filtered_docs = []

        for doc in documents:
            filtered_relations = [r for r in doc.relations if r.relation_type in relation_types]
            if filtered_relations:  # Only keep documents with relevant relations
                filtered_doc = Document(
                    pmid=doc.pmid,
                    title=doc.title,
                    abstract=doc.abstract,
                    entities=doc.entities,
                    relations=filtered_relations
                )
                filtered_docs.append(filtered_doc)

        return filtered_docs

    def get_entity_types_in_relations(self, documents: List[Document]) -> Dict[str, Dict[str, int]]:
        """Get statistics about entity types involved in relations."""
        relation_entity_stats = {}

        for doc in documents:
            # Create entity lookup by ID
            entity_lookup = {}
            for entity in doc.entities:
                for entity_id in entity.entity_id.split(','):
                    entity_lookup[entity_id.strip()] = entity

            for relation in doc.relations:
                entity1 = entity_lookup.get(relation.entity1_id)
                entity2 = entity_lookup.get(relation.entity2_id)

                if entity1 and entity2:
                    rel_type = relation.relation_type
                    if rel_type not in relation_entity_stats:
                        relation_entity_stats[rel_type] = {}

                    pair_key = f"{entity1.entity_type}--{entity2.entity_type}"
                    relation_entity_stats[rel_type][pair_key] = relation_entity_stats[rel_type].get(pair_key, 0) + 1

        return relation_entity_stats
