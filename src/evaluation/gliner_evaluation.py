import json
import os
from typing import Dict, List, Tuple, Set
from datetime import datetime
from tqdm import tqdm
from gliner import GLiNER
from biored_data_loader import BioREDDataLoader, Document, Entity


class GLiNERExtractor:
    """GLiNER-based Named Entity Recognition extractor for biomedical texts."""

    def __init__(self, model_name: str = "Ihor/gliner-biomed-bi-large-v1.0", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        self.labels = [
            "GeneOrGeneProduct",
            "DiseaseOrPhenotypicFeature",
            "ChemicalEntity",
            "SequenceVariant",
            "CellLine",
            "OrganismTaxon"
        ]

    def load_model(self):
        """Load the GLiNER model."""
        print(f"Loading GLiNER model: {self.model_name}")
        try:
            self.model = GLiNER.from_pretrained(self.model_name)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def extract_entities_from_document(self, document: Document) -> List[Dict]:
        """Extract entities from a document using GLiNER."""
        if self.model is None:
            self.load_model()

        # Combine title and abstract
        text = f"{document.title} {document.abstract}"

        # Extract entities
        predictions = self.model.predict_entities(
            text,
            labels=self.labels,
            threshold=self.threshold
        )

        # Convert predictions to standard format
        extracted_entities = []
        for pred in predictions:
            entity = {
                'text': pred['text'],
                'start': pred['start'],
                'end': pred['end'],
                'entity_type': pred['label'],
                'score': pred['score']
            }
            extracted_entities.append(entity)

        return extracted_entities


class NERMetrics:
    """Calculate NER-specific metrics."""

    @staticmethod
    def exact_match(pred_entities: List[Dict], gold_entities: List[Entity]) -> Tuple[Set, Set, Set]:
        """Calculate exact span matches between predicted and gold entities."""
        pred_set = {(e['start'], e['end'], e['entity_type']) for e in pred_entities}
        gold_set = {(e.start, e.end, e.entity_type) for e in gold_entities}

        true_positives = pred_set & gold_set
        false_positives = pred_set - gold_set
        false_negatives = gold_set - pred_set

        return true_positives, false_positives, false_negatives

    @staticmethod
    def partial_match(pred_entities: List[Dict], gold_entities: List[Entity],
                      overlap_threshold: float = 0.5) -> Tuple[Set, Set, Set]:
        """Calculate partial matches based on overlap threshold."""
        matched_pred = set()
        matched_gold = set()

        for i, pred in enumerate(pred_entities):
            for j, gold in enumerate(gold_entities):
                if pred['entity_type'] == gold.entity_type:
                    # Calculate overlap
                    overlap_start = max(pred['start'], gold.start)
                    overlap_end = min(pred['end'], gold.end)

                    if overlap_start < overlap_end:
                        overlap_len = overlap_end - overlap_start
                        pred_len = pred['end'] - pred['start']
                        gold_len = gold.end - gold.start

                        # IoU calculation
                        union_len = pred_len + gold_len - overlap_len
                        iou = overlap_len / union_len if union_len > 0 else 0

                        if iou >= overlap_threshold:
                            matched_pred.add(i)
                            matched_gold.add(j)

        true_positives = len(matched_pred)
        false_positives = len(pred_entities) - len(matched_pred)
        false_negatives = len(gold_entities) - len(matched_gold)

        return true_positives, false_positives, false_negatives

    @staticmethod
    def text_match(pred_entities: List[Dict], gold_entities: List[Entity]) -> Tuple[Set, Set, Set]:
        """Calculate matches based on text content only (ignoring spans)."""
        pred_set = {(e['text'].lower(), e['entity_type']) for e in pred_entities}
        gold_set = {(e.text.lower(), e.entity_type) for e in gold_entities}

        true_positives = pred_set & gold_set
        false_positives = pred_set - gold_set
        false_negatives = gold_set - pred_set

        return true_positives, false_positives, false_negatives

    @staticmethod
    def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }


class GLiNEREvaluationFramework:
    """Framework for evaluating GLiNER models on BioRED dataset."""

    def __init__(self, data_path: str, results_path: str = "results/gliner_evaluation"):
        self.data_path = data_path
        self.results_path = results_path
        self.data_loader = BioREDDataLoader(data_path)
        self.extractors = {}
        self.metrics = NERMetrics()

        # Create results directory
        os.makedirs(results_path, exist_ok=True)

        # Load datasets
        self.train_docs = self.data_loader.load_training_data()
        self.dev_docs = self.data_loader.load_development_data()
        self.test_docs = self.data_loader.load_test_data()

        print("Dataset loaded:")
        print(f"  Training: {len(self.train_docs)} documents")
        print(f"  Development: {len(self.dev_docs)} documents")
        print(f"  Test: {len(self.test_docs)} documents")

    def add_model(self, name: str, model_path: str, threshold: float = 0.5):
        """Add a model configuration for evaluation."""
        self.extractors[name] = {
            'model_path': model_path,
            'threshold': threshold,
            'extractor': None
        }

    def initialize_model(self, name: str):
        """Initialize a specific model."""
        if name in self.extractors and self.extractors[name]['extractor'] is None:
            config = self.extractors[name]
            extractor = GLiNERExtractor(
                model_name=config['model_path'],
                threshold=config['threshold']
            )
            extractor.load_model()
            self.extractors[name]['extractor'] = extractor

    def evaluate_model(self, model_name: str, documents: List[Document],
                       max_docs: int = None, matching_mode: str = 'exact'):
        """Evaluate model with progress tracking."""

        if max_docs:
            documents = documents[:max_docs]

        if model_name not in self.extractors:
            raise ValueError(f"Model {model_name} not found")

        # Initialize model if needed
        if self.extractors[model_name]['extractor'] is None:
            self.initialize_model(model_name)

        extractor = self.extractors[model_name]['extractor']

        all_predictions = []
        all_ground_truth = []
        document_results = []

        print(f"\nProcessing {len(documents)} documents...")

        # Process documents with progress bar
        with tqdm(total=len(documents), desc=f"Evaluating {model_name}", unit="docs") as pbar:
            for doc in documents:
                try:
                    # Extract entities
                    pred_entities = extractor.extract_entities_from_document(doc)

                    # Calculate metrics based on matching mode
                    if matching_mode == 'exact':
                        tp, fp, fn = self.metrics.exact_match(pred_entities, doc.entities)
                    elif matching_mode == 'partial':
                        tp, fp, fn = self.metrics.partial_match(pred_entities, doc.entities)
                    elif matching_mode == 'text':
                        tp, fp, fn = self.metrics.text_match(pred_entities, doc.entities)
                    else:
                        raise ValueError(f"Unknown matching mode: {matching_mode}")

                    # Store results
                    doc_result = {
                        'pmid': doc.pmid,
                        'predictions': pred_entities,
                        'ground_truth': [
                            {
                                'text': e.text,
                                'start': e.start,
                                'end': e.end,
                                'entity_type': e.entity_type
                            } for e in doc.entities
                        ],
                        'metrics': self.metrics.calculate_metrics(
                            len(tp) if isinstance(tp, set) else tp,
                            len(fp) if isinstance(fp, set) else fp,
                            len(fn) if isinstance(fn, set) else fn
                        )
                    }
                    document_results.append(doc_result)

                    all_predictions.extend(pred_entities)
                    all_ground_truth.extend(doc.entities)

                except Exception as e:
                    print(f"\nError processing document {doc.pmid}: {e}")

                pbar.update(1)

        return {
            'model_name': model_name,
            'total_documents': len(documents),
            'document_results': document_results,
            'predictions': all_predictions,
            'ground_truth': all_ground_truth,
            'matching_mode': matching_mode
        }

    def analyze_entity_type_performance(self, results: Dict):
        """
        Provides a detailed breakdown of how well the GLiNER model performs on each specific entity type.

        Calculates precision, recall, and F1-score for each individual entity type (like "GeneOrGeneProduct",
        "DiseaseOrPhenotypicFeature", "ChemicalEntity", etc.) rather than just overall performance.

        Process:
        1. Groups entities by type: Separates predictions and ground truth entities by their entity types
        2. Calculates per-type metrics: For each entity type, filters predictions and ground truth to only
           that type, applies the same matching mode (exact/partial/text) used in the main evaluation,
           and computes precision, recall, and F1-score for that specific type

        This helps identify which entity types the model struggles with most, allowing for targeted
        improvements or understanding of model limitations.
        """
        print("\n=== Entity Type Performance Analysis ===")

        model_name = results['model_name']
        print(f"\n{model_name}:")

        # Group by entity type
        type_metrics = {}

        for doc_result in results['document_results']:
            for pred in doc_result['predictions']:
                etype = pred['entity_type']
                if etype not in type_metrics:
                    type_metrics[etype] = {'tp': 0, 'fp': 0, 'fn': 0}

            for gt in doc_result['ground_truth']:
                etype = gt['entity_type']
                if etype not in type_metrics:
                    type_metrics[etype] = {'tp': 0, 'fp': 0, 'fn': 0}

        # Recalculate metrics per type
        for doc_result in results['document_results']:
            pred_entities = doc_result['predictions']
            gold_entities = [Entity(
                pmid=doc_result['pmid'],
                start=g['start'],
                end=g['end'],
                text=g['text'],
                entity_type=g['entity_type'],
                entity_id=""
            ) for g in doc_result['ground_truth']]

            # Calculate per-type metrics
            for etype in type_metrics:
                pred_filtered = [p for p in pred_entities if p['entity_type'] == etype]
                gold_filtered = [g for g in gold_entities if g.entity_type == etype]

                if results['matching_mode'] == 'exact':
                    tp, fp, fn = self.metrics.exact_match(pred_filtered, gold_filtered)
                    tp_count = len(tp)
                    fp_count = len(fp)
                    fn_count = len(fn)
                else:
                    tp_count, fp_count, fn_count = self.metrics.partial_match(
                        pred_filtered, gold_filtered
                    )

                type_metrics[etype]['tp'] += tp_count
                type_metrics[etype]['fp'] += fp_count
                type_metrics[etype]['fn'] += fn_count

        # Calculate and display metrics
        for etype, counts in sorted(type_metrics.items()):
            metrics = self.metrics.calculate_metrics(
                counts['tp'], counts['fp'], counts['fn']
            )
            print(f"\n  {etype}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")
            print(f"    Support: {counts['tp'] + counts['fn']}")

    def calculate_overall_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate overall metrics across all documents."""
        total_tp = sum(doc['metrics']['true_positives'] for doc in results['document_results'])
        total_fp = sum(doc['metrics']['false_positives'] for doc in results['document_results'])
        total_fn = sum(doc['metrics']['false_negatives'] for doc in results['document_results'])

        return self.metrics.calculate_metrics(total_tp, total_fp, total_fn)

    def save_results(self, results: Dict, filename: str):
        """Save evaluation results to JSON file."""
        # Convert to JSON-serializable format
        json_results = {
            'model_name': results['model_name'],
            'total_documents': results['total_documents'],
            'matching_mode': results['matching_mode'],
            'overall_metrics': self.calculate_overall_metrics(results),
            'num_predictions': len(results['predictions']),
            'num_ground_truth': len(results['ground_truth']),
            'timestamp': datetime.now().isoformat()
        }

        output_path = os.path.join(self.results_path, filename)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {output_path}")

    def generate_evaluation_report(self, all_results: Dict[str, Dict]):
        """Generate a comprehensive evaluation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_path, f"evaluation_report_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write("BioRED Named Entity Recognition Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Dataset Statistics:\n")
            f.write(f"  Training documents: {len(self.train_docs)}\n")
            f.write(f"  Development documents: {len(self.dev_docs)}\n")
            f.write(f"  Test documents: {len(self.test_docs)}\n\n")

            # Results for each model and matching mode
            for key, results in all_results.items():
                if results:
                    model_name = results['model_name']
                    matching_mode = results['matching_mode']
                    overall_metrics = self.calculate_overall_metrics(results)

                    f.write(f"\nModel: {model_name} (Matching: {matching_mode})\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Documents processed: {results['total_documents']}\n")
                    f.write(f"  Total predictions: {len(results['predictions'])}\n")
                    f.write(f"  Total ground truth: {len(results['ground_truth'])}\n")
                    f.write(f"  Precision: {overall_metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {overall_metrics['recall']:.4f}\n")
                    f.write(f"  F1 Score: {overall_metrics['f1']:.4f}\n")
                    f.write(f"  True Positives: {overall_metrics['true_positives']}\n")
                    f.write(f"  False Positives: {overall_metrics['false_positives']}\n")
                    f.write(f"  False Negatives: {overall_metrics['false_negatives']}\n")

        print(f"Evaluation report saved to {report_path}")

    def print_evaluation_summary(self, all_results: Dict[str, Dict]):
        """Print formatted summary of all evaluations."""
        print("\n" + "="*80)
        print("NER EVALUATION SUMMARY")
        print("="*80)
        print(f"{'Model':<30} {'Matching':<10} {'F1':>8} {'Precision':>10} {'Recall':>10}")
        print("-"*80)

        # Sort by F1 score
        sorted_results = []
        for key, results in all_results.items():
            if results:
                metrics = self.calculate_overall_metrics(results)
                sorted_results.append((
                    results['model_name'],
                    results['matching_mode'],
                    metrics
                ))

        sorted_results.sort(key=lambda x: x[2]['f1'], reverse=True)

        for model_name, matching_mode, metrics in sorted_results:
            print(f"{model_name:<30} {matching_mode:<10} "
                  f"{metrics['f1']:>8.3f} {metrics['precision']:>10.3f} "
                  f"{metrics['recall']:>10.3f}")

        print("="*80)

        # Best model
        if sorted_results:
            best_model, best_mode, best_metrics = sorted_results[0]
            print(f"\nBest performing configuration: {best_model} ({best_mode} matching)")
            print(f"F1 Score: {best_metrics['f1']:.3f}")

    def run_complete_evaluation(self):
        """Run complete evaluation with multiple matching modes."""

        # Define evaluation parameters
        EVAL_DOCS = 100  # Number of documents to evaluate
        MATCHING_MODES = ['exact', 'partial', 'text']

        print("=" * 50)
        print("Starting GLiNER NER Evaluation")
        print(f"Results will be saved in: {self.results_path}")
        print("=" * 50 + "\n")

        all_results = {}

        # Evaluate each model with different matching modes
        for model_name in tqdm(self.extractors.keys(), desc="Models", unit="model"):
            for matching_mode in MATCHING_MODES:
                try:
                    tqdm.write(f"\nEvaluating {model_name} with {matching_mode} matching...")

                    result = self.evaluate_model(
                        model_name,
                        self.test_docs,
                        max_docs=EVAL_DOCS,
                        matching_mode=matching_mode
                    )

                    key = f"{model_name}_{matching_mode}"
                    all_results[key] = result

                    # Show metrics
                    metrics = self.calculate_overall_metrics(result)
                    tqdm.write(f"✓ {model_name} ({matching_mode}): "
                               f"F1={metrics['f1']:.3f}, "
                               f"P={metrics['precision']:.3f}, "
                               f"R={metrics['recall']:.3f}")

                    # Analyze entity type performance
                    self.analyze_entity_type_performance(result)

                    # Save individual results
                    self.save_results(result, f"{key}_results.json")

                except Exception as e:
                    tqdm.write(f"✗ Failed to evaluate {model_name} ({matching_mode}): {e}")
                    all_results[f"{model_name}_{matching_mode}"] = None

        print("\n" + "="*50)

        if all_results:
            # Generate final report
            print("\nGenerating evaluation report...")
            self.generate_evaluation_report(all_results)

            # Print summary
            self.print_evaluation_summary(all_results)
        else:
            print("ERROR: No successful evaluations to report")

        print("\n" + "="*50)
        print("Complete evaluation finished!")
        print(f"Results saved in: {self.results_path}")


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "evaluation/data/BioRED"
    RESULTS_PATH = "evaluation/results/gliner_evaluation"

    # Initialize framework
    print("Initializing GLiNER Evaluation Framework...")
    framework = GLiNEREvaluationFramework(
        data_path=DATA_PATH,
        results_path=RESULTS_PATH
    )

    # Setup models with different configurations
    print("\nSetting up model configurations...")
    framework.add_model(
        name="gliner_biomed_default",
        model_path="Ihor/gliner-biomed-bi-large-v1.0",
        threshold=0.5
    )

    framework.add_model(
        name="gliner_biomed_low_threshold",
        model_path="Ihor/gliner-biomed-bi-large-v1.0",
        threshold=0.3
    )

    framework.add_model(
        name="gliner_biomed_high_threshold",
        model_path="Ihor/gliner-biomed-bi-large-v1.0",
        threshold=0.7
    )

    print("Model configurations added")

    # Run evaluation
    print("\nStarting evaluation...")
    try:
        framework.run_complete_evaluation()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
