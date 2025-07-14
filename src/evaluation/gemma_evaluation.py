import json
import os
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
from biored_data_loader import BioREDDataLoader, Document
from gemma_relation_extractor import GemmaEvaluator
from relation_metrics import RelationMetrics


class GemmaEvaluationFramework:
    """Framework for evaluating Gemma and MedGemma models on BioRED dataset."""
    def __init__(self, data_path: str, results_path: str = "results"):
        self.data_path = data_path
        self.results_path = results_path
        self.data_loader = BioREDDataLoader(data_path)
        self.evaluator = GemmaEvaluator(self.data_loader)

        # Create results directory
        os.makedirs(results_path, exist_ok=True)

        # Load datasets
        self.train_docs = self.data_loader.load_training_data()
        self.dev_docs = self.data_loader.load_development_data()
        self.test_docs = self.data_loader.load_test_data()

        print("Load dataset:")
        print(f"  Training: {len(self.train_docs)} documents")
        print(f"  Development: {len(self.dev_docs)} documents")
        print(f"  Test: {len(self.test_docs)} documents")

    def setup_models(self):
        """Setup Gemma and MedGemma models for evaluation."""
        # Standard Gemma model
        self.evaluator.add_model(
            name="gemma_basic",
            model_path="google/gemma-3-4b-it",
            prompt_strategy="basic"
        )

        self.evaluator.add_model(
            name="gemma_few_shot",
            model_path="google/gemma-3-4b-it",
            prompt_strategy="few_shot"
        )

        self.evaluator.add_model(
            name="gemma_structured",
            model_path="google/gemma-3-4b-it",
            prompt_strategy="structured"
        )

        self.evaluator.add_model(
            name="medgemma_basic",
            model_path="google/medgemma-4b-it",
            prompt_strategy="basic"
        )

        self.evaluator.add_model(
            name="medgemma_few_shot",
            model_path="google/medgemma-4b-it",
            prompt_strategy="few_shot"
        )

        self.evaluator.add_model(
            name="medgemma_structured",
            model_path="google/medgemma-4b-it",
            prompt_strategy="structured"
        )
        print("Models setup complete:")

    def evaluate_model_with_batching(self, model_name: str, documents: List[Document],
                                     batch_size: int = 2, max_docs: int = None,
                                     has_entity_context: bool = True):
        """Evaluate model with batch processing and progress tracking."""

        if max_docs:
            documents = documents[:max_docs]

        total_batches = (len(documents) + batch_size - 1) // batch_size
        all_predictions = []
        all_ground_truth = []

        print(f"\nProcessing {len(documents)} documents in {total_batches} batches...")

        # Progress bar for batches
        with tqdm(total=len(documents), desc=f"Evaluating {model_name}", unit="docs") as pbar:
            for batch_idx in range(0, len(documents), batch_size):
                batch_docs = documents[batch_idx:batch_idx + batch_size]

                try:
                    # Initialize model if needed
                    if self.evaluator.models[model_name]['extractor'] is None:
                        self.evaluator.initialize_model(model_name)

                    extractor = self.evaluator.models[model_name]['extractor']

                    # Process each document in batch
                    for doc in batch_docs:
                        # Get predictions
                        predictions = extractor.extract_relations_from_document(doc, has_entity_context)

                        # Convert predictions
                        for pred in predictions:
                            all_predictions.append((
                                doc.pmid,
                                pred['entity1'],
                                pred['entity2'],
                                pred['relation']
                            ))

                        # Get ground truth
                        entity_lookup = {}
                        for entity in doc.entities:
                            for entity_id in entity.entity_id.split(','):
                                entity_lookup[entity_id.strip()] = entity

                        for relation in doc.relations:
                            entity1 = entity_lookup.get(relation.entity1_id)
                            entity2 = entity_lookup.get(relation.entity2_id)

                            if entity1 and entity2:
                                all_ground_truth.append((
                                    doc.pmid,
                                    entity1.text,
                                    entity2.text,
                                    relation.relation_type
                                ))

                        pbar.update(1)

                except Exception as e:
                    print(f"\nError in batch starting at index {batch_idx}: {e}")
                    pbar.update(len(batch_docs))  # Update progress even on error
                    continue

        return {
            'model_name': model_name,
            'total_documents': len(documents),
            'predictions': all_predictions,
            'ground_truth': all_ground_truth
        }

    def analyze_relation_specific_performance(self, results: Dict):
        """Analyze performance by relation type."""
        print("\n=== Relation-Specific Performance Analysis ===")

        for model_name, result in results.items():
            if result is None:
                continue

            print(f"\n{model_name}:")

            # Group predictions and ground truth by relation type
            pred_by_relation = {}
            gt_by_relation = {}

            for pred in result['predictions']:
                rel_type = pred[3]  # relation type is 4th element
                if rel_type not in pred_by_relation:
                    pred_by_relation[rel_type] = []
                pred_by_relation[rel_type].append(pred)

            for gt in result['ground_truth']:
                rel_type = gt[3]
                if rel_type not in gt_by_relation:
                    gt_by_relation[rel_type] = []
                gt_by_relation[rel_type].append(gt)

            # Calculate metrics per relation type
            all_relation_types = set(pred_by_relation.keys()) | set(gt_by_relation.keys())

            for rel_type in sorted(all_relation_types):
                preds = pred_by_relation.get(rel_type, [])
                gts = gt_by_relation.get(rel_type, [])

                if preds or gts:
                    metrics = RelationMetrics()
                    # Use normalized matching with bidirectional for relation-specific analysis
                    metrics.add_predictions(preds, gts, matching_mode='normalized', handle_bidirectional=True)
                    rel_metrics = metrics.calculate_metrics()

                    print(f"  {rel_type}:")
                    print(f"    Precision: {rel_metrics['precision']:.4f}")
                    print(f"    Recall: {rel_metrics['recall']:.4f}")
                    print(f"    F1 Score: {rel_metrics['f1']:.4f}")
                    print(f"    Predictions: {len(preds)}, Ground Truth: {len(gts)}")

    # def aggregate_all_results(self, *result_dicts):
    #     """Safely aggregate results from multiple evaluations."""
    #     all_results = {}

    #     for result_dict in result_dicts:
    #         if result_dict:  # Check if not None
    #             for model_name, result in result_dict.items():
    #                 if result is not None:  # Check individual results
    #                     all_results[model_name] = result
    #                 else:
    #                     print(f"Skipping null result for {model_name}")

    #     return all_results

    def print_evaluation_summary(self, all_results):
        """Print formatted summary of all evaluations."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>10} {'Docs':>8}")
        print("-"*70)

        # Sort by F1 score
        sorted_results = []
        for model_name, result in all_results.items():
            if result:
                metrics = RelationMetrics()
                # Use partial matching with bidirectional handling
                metrics.add_predictions(result['predictions'], result['ground_truth'],
                                        matching_mode='partial', handle_bidirectional=True)
                m = metrics.calculate_metrics()
                sorted_results.append((model_name, m, result['total_documents']))

        sorted_results.sort(key=lambda x: x[1]['f1'], reverse=True)

        for model_name, m, total_docs in sorted_results:
            print(f"{model_name:<25} {m['f1']:>8.3f} {m['precision']:>10.3f} {m['recall']:>10.3f} {total_docs:>8}")

        print("="*70)

        # Best model
        if sorted_results:
            best_model, best_metrics, _ = sorted_results[0]
            print(f"\nBest performing model: {best_model} (F1: {best_metrics['f1']:.3f})")

            # Check against thesis targets
            if best_metrics['f1'] >= 0.50:
                print("✓ Achieved target F1 score (≥ 0.50)")
            elif best_metrics['f1'] >= 0.40:
                print("✓ Achieved minimum viable F1 score (≥ 0.40)")
            else:
                print("✗ Below minimum viable F1 score (< 0.40)")

    def save_results(self, results: Dict, filename: str):
        """Save evaluation results to JSON file."""
        # Convert results to JSON-serializable format
        json_results = {}

        for model_name, result in results.items():
            if result:
                json_results[model_name] = {
                    'model_name': result['model_name'],
                    'total_documents': result['total_documents'],
                    'num_predictions': len(result['predictions']),
                    'num_ground_truth': len(result['ground_truth']),
                    'predictions': result['predictions'],
                    'ground_truth': result['ground_truth']
                }

        output_path = os.path.join(self.results_path, filename)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {output_path}")

    def generate_evaluation_report(self, results: Dict):
        """Generate a comprehensive evaluation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_path, f"evaluation_report_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write("BioRED Relation Extraction Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Dataset Statistics:\n")
            f.write(f"  Training documents: {len(self.train_docs)}\n")
            f.write(f"  Development documents: {len(self.dev_docs)}\n")
            f.write(f"  Test documents: {len(self.test_docs)}\n\n")

            for model_name, result in results.items():
                if result:
                    f.write(f"Model: {model_name}\n")
                    f.write("-" * 30 + "\n")

                    metrics = RelationMetrics()
                    metrics.add_predictions(result['predictions'], result['ground_truth'],
                                          matching_mode='partial', handle_bidirectional=True)
                    model_metrics = metrics.calculate_metrics()

                    f.write(f"  Documents processed: {result['total_documents']}\n")
                    f.write(f"  Predictions: {len(result['predictions'])}\n")
                    f.write(f"  Ground truth: {len(result['ground_truth'])}\n")
                    f.write(f"  Precision: {model_metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {model_metrics['recall']:.4f}\n")
                    f.write(f"  F1 Score: {model_metrics['f1']:.4f}\n")
                    f.write(f"  True Positives: {model_metrics['true_positives']}\n")
                    f.write(f"  False Positives: {model_metrics['false_positives']}\n")
                    f.write(f"  False Negatives: {model_metrics['false_negatives']}\n\n")

        print(f"Evaluation report saved to {report_path}")

    def run_complete_evaluation(self):
        """Run complete evaluation with progress tracking."""

        # Define evaluation parameters
        MODEL_COMPARE_DOCS = 100
        BATCH_SIZE = 2
        counter = 0

        print("=" * 50)
        print(f"Results will be saved in: {self.results_path}\n")
        print("\n" + "="*50)

        model_pairs = [
            ("gemma_structured", "medgemma_structured"),
            ("gemma_basic", "medgemma_basic"),
            ("gemma_few_shot", "medgemma_few_shot")
        ]

        model_comparison_results = {}

        for gemma_model, medgemma_model in tqdm(model_pairs, desc="Model comparisons", unit="pair"):
            for model_name in [gemma_model, medgemma_model]:
                if model_name in self.evaluator.models:
                    counter += 1
                    # has_entity_context = (counter % 2 == 0)  # Alternate between True and False
                    try:
                        tqdm.write(f"\nEvaluating {model_name}...")
                        result = self.evaluate_model_with_batching(
                            model_name,
                            self.test_docs,
                            batch_size=BATCH_SIZE,
                            max_docs=MODEL_COMPARE_DOCS,
                            has_entity_context=True
                        )
                        model_comparison_results[model_name] = result

                        # Show metrics
                        metrics = RelationMetrics()
                        metrics.add_predictions(result['predictions'], result['ground_truth'],
                                                matching_mode='partial', handle_bidirectional=True)
                        m = metrics.calculate_metrics()
                        tqdm.write(f"✓ {model_name}: F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}")

                    except Exception as e:
                        tqdm.write(f"✗ Failed to evaluate {model_name}: {e}")
                        model_comparison_results[model_name] = None

        if model_comparison_results:
            self.save_results(model_comparison_results, "model_comparison_results.json")

        print("\n" + "="*50)

        if model_comparison_results:
            # Analyze with progress bar
            print("\nAnalyzing relation-specific performance...")
            self.analyze_relation_specific_performance(model_comparison_results)

            # Generate final report
            print("\nGenerating evaluation report...")
            self.generate_evaluation_report(model_comparison_results)

            # Print summary
            self.print_evaluation_summary(model_comparison_results)
        else:
            print("ERROR: No successful evaluations to report")

        print("\n" + "=" * 50)
        print("Complete evaluation finished!")
        print(f"Results saved in: {self.results_path}")


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "evaluation/data/BioRED"
    RESULTS_PATH = "evaluation/results/gemma_evaluation"

    # Initialize framework
    print("Initializing Gemma Evaluation Framework...")
    framework = GemmaEvaluationFramework(
        data_path=DATA_PATH,
        results_path=RESULTS_PATH
    )

    # Check model availability with progress
    print("\nChecking model availability...")
    all_models = [
        "gemma_basic", "gemma_few_shot", "gemma_structured",
        "medgemma_basic", "medgemma_few_shot", "medgemma_structured"
    ]

    available_models = []
    framework.setup_models()

    for model_name in tqdm(all_models, desc="Checking models", unit="model"):
        try:
            framework.evaluator.initialize_model(model_name)
            available_models.append(model_name)
            tqdm.write(f"✓ {model_name} available")
        except Exception as e:
            tqdm.write(f"✗ {model_name} not available: {str(e)[:50]}...")

    print(f"\nFound {len(available_models)} available models out of {len(all_models)}")

    if available_models:
        print("\nStarting evaluation...")
        try:
            framework.run_complete_evaluation()
        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user")
        except Exception as e:
            print(f"\nEvaluation failed with error: {e}")
    else:
        print("\nNo models available for evaluation")
        print("Please ensure you have installed mlx-lm and have access to model files")
        print("Installation: pip install mlx-lm")
        print("Model access: Ensure Gemma/MedGemma models are available")
