#!/usr/bin/env python3
"""
GLiNER Entity Recognition Evaluation Framework
Evaluates GLiNER performance on i2b2 2010 dataset for thesis research
"""

import json
import os
import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import GLiNER - ensure it's available
try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    print("Warning: GLiNER not available. Install with: pip install gliner")
    GLINER_AVAILABLE = False


@dataclass
class Entity:
    """Represents a named entity with its span and type."""
    start: int
    end: int
    text: str
    entity_type: str
    
    def __hash__(self):
        return hash((self.start, self.end, self.text, self.entity_type))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.start == other.start and 
                self.end == other.end and 
                self.text == other.text and 
                self.entity_type == other.entity_type)
    
    def overlaps_with(self, other: 'Entity', min_overlap: float = 0.5) -> bool:
        """Check if this entity overlaps with another entity."""
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        
        if overlap_start >= overlap_end:
            return False
        
        overlap_length = overlap_end - overlap_start
        min_length = min(self.end - self.start, other.end - other.start)
        
        return overlap_length / min_length >= min_overlap


class GLiNEREvaluator:
    """Evaluates GLiNER model performance on i2b2 2010 NER task."""
    
    def __init__(self, model_path: str = "Ihor/gliner-biomed-bi-large-v1.0"):
        """Initialize the evaluator with a GLiNER model."""
        if not GLINER_AVAILABLE:
            raise ImportError("GLiNER not available. Install with: pip install gliner")
        
        print(f"Loading GLiNER model: {model_path}")
        self.model = GLiNER.from_pretrained(model_path)
        
        # GLiNER entity labels - biomedical focus
        self.entity_labels = [
            "Disease or Condition", "Medication", "Medication Dosage and Frequency",
            "Procedure", "Lab Test", "Lab Test Result", "Body Site",
            "Medical Device", "Demographic Information"
        ]
        
        print(f"GLiNER model loaded with {len(self.entity_labels)} entity types")
    
    def predict_entities(self, text: str, threshold: float = 0.5) -> List[Entity]:
        """Predict entities in text using GLiNER."""
        if not GLINER_AVAILABLE:
            return []
        
        predictions = self.model.predict_entities(
            text, 
            labels=self.entity_labels, 
            threshold=threshold
        )
        
        entities = []
        for pred in predictions:
            entity = Entity(
                start=pred['start'],
                end=pred['end'],
                text=pred['text'],
                entity_type=pred['label']
            )
            entities.append(entity)
        
        return entities
    
    def load_i2b2_2010_data(self, data_dir: str) -> List[Dict]:
        """Load i2b2 2010 concept extraction data."""
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"i2b2 data directory not found: {data_dir}")
        
        samples = []
        
        # Expected directory structure:
        # data_dir/
        #   train/
        #     *.txt (text files)
        #     *.con (concept annotation files)
        #   test/
        #     *.txt (text files)
        #     *.con (concept annotation files)
        
        for split in ['train', 'test']:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                print(f"Warning: {split} directory not found: {split_dir}")
                continue
            
            txt_files = [f for f in os.listdir(split_dir) if f.endswith('.txt')]
            print(f"Found {len(txt_files)} text files in {split} split")
            
            for txt_file in txt_files:
                base_name = txt_file.replace('.txt', '')
                txt_path = os.path.join(split_dir, txt_file)
                con_path = os.path.join(split_dir, base_name + '.con')
                
                if not os.path.exists(con_path):
                    print(f"Warning: Missing annotation file: {con_path}")
                    continue
                
                try:
                    # Read text
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Read concept annotations
                    entities = self._parse_i2b2_concepts(con_path, text)
                    
                    samples.append({
                        'id': base_name,
                        'text': text,
                        'entities': entities,
                        'split': split
                    })
                    
                except Exception as e:
                    print(f"Error processing {txt_file}: {e}")
                    continue
        
        print(f"Loaded {len(samples)} samples total")
        return samples
    
    def _parse_i2b2_concepts(self, con_path: str, text: str) -> List[Entity]:
        """Parse i2b2 concept annotation file."""
        entities = []
        
        with open(con_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # i2b2 format: c="concept" 123:45 123:56||t="type"
                    # Updated regex to handle more flexible whitespace
                    match = re.match(r'c="([^"]+)"\s+(\d+):(\d+)\s+(\d+):(\d+)\s*\|\|t="([^"]+)"', line)
                    if match:
                        concept, start1, end1, start2, end2, concept_type = match.groups()
                        
                        # Convert to GLiNER entity types
                        gliner_type = self._map_i2b2_to_gliner_type(concept_type)
                        
                        entity = Entity(
                            start=int(start1),
                            end=int(end2),
                            text=concept,
                            entity_type=gliner_type
                        )
                        entities.append(entity)
                    else:
                        print(f"Warning: Could not parse line {line_num} in {con_path}: {line}")
                        
                except Exception as e:
                    print(f"Error parsing line {line_num} in {con_path}: {e}")
                    continue
        
        return entities
    
    def _map_i2b2_to_gliner_type(self, i2b2_type: str) -> str:
        """Map i2b2 concept types to GLiNER entity types."""
        mapping = {
            'problem': 'Disease or Condition',
            'treatment': 'Medication',
            'test': 'Lab Test',
            'procedure': 'Procedure',
            'body_part': 'Body Site',
            'medication': 'Medication',
            'device': 'Medical Device',
            'drug': 'Medication',
            'symptom': 'Disease or Condition',
            'disease': 'Disease or Condition'
        }
        
        # Normalize the type
        normalized_type = i2b2_type.lower().strip()
        return mapping.get(normalized_type, 'Disease or Condition')
    
    def evaluate_strict_matching(self, predictions: List[Entity], gold_entities: List[Entity]) -> Dict:
        """Evaluate with strict matching (exact span and type match)."""
        pred_set = set(predictions)
        gold_set = set(gold_entities)
        
        true_positives = pred_set & gold_set
        false_positives = pred_set - gold_set
        false_negatives = gold_set - pred_set
        
        precision = len(true_positives) / len(pred_set) if pred_set else 0.0
        recall = len(true_positives) / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'support': len(gold_set)
        }
    
    def evaluate_relaxed_matching(self, predictions: List[Entity], gold_entities: List[Entity]) -> Dict:
        """Evaluate with relaxed matching (overlap-based)."""
        matched_pred = set()
        matched_gold = set()
        
        for pred in predictions:
            for gold in gold_entities:
                if pred.overlaps_with(gold) and pred.entity_type == gold.entity_type:
                    matched_pred.add(pred)
                    matched_gold.add(gold)
                    break
        
        true_positives = len(matched_pred)
        false_positives = len(predictions) - true_positives
        false_negatives = len(gold_entities) - len(matched_gold)
        
        precision = true_positives / len(predictions) if predictions else 0.0
        recall = true_positives / len(gold_entities) if gold_entities else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'support': len(gold_entities)
        }
    
    def evaluate_by_entity_type(self, predictions: List[Entity], gold_entities: List[Entity]) -> Dict:
        """Evaluate performance by entity type."""
        results = {}
        
        # Group entities by type
        pred_by_type = defaultdict(list)
        gold_by_type = defaultdict(list)
        
        for pred in predictions:
            pred_by_type[pred.entity_type].append(pred)
        
        for gold in gold_entities:
            gold_by_type[gold.entity_type].append(gold)
        
        # Get all entity types
        all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())
        
        for entity_type in all_types:
            preds = pred_by_type[entity_type]
            golds = gold_by_type[entity_type]
            
            strict_results = self.evaluate_strict_matching(preds, golds)
            relaxed_results = self.evaluate_relaxed_matching(preds, golds)
            
            results[entity_type] = {
                'strict': strict_results,
                'relaxed': relaxed_results,
                'predicted_count': len(preds),
                'gold_count': len(golds)
            }
        
        return results
    
    def analyze_threshold_sensitivity(self, dataset: List[Dict], thresholds: List[float] = None) -> Dict:
        """Analyze performance at different confidence thresholds."""
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = {}
        
        for threshold in thresholds:
            print(f"Evaluating at threshold: {threshold}")
            threshold_results = self.run_evaluation(dataset, threshold)
            results[threshold] = threshold_results['overall']
        
        return results
    
    def run_evaluation(self, dataset: List[Dict], threshold: float = 0.5) -> Dict:
        """Run complete evaluation on a dataset."""
        all_predictions = []
        all_gold = []
        sample_results = []
        
        print(f"Running evaluation on {len(dataset)} samples with threshold {threshold}")
        
        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{len(dataset)}")
            
            text = sample['text']
            gold_entities = sample['entities']
            
            # Predict entities
            predictions = self.predict_entities(text, threshold)
            
            # Evaluate this sample
            strict_results = self.evaluate_strict_matching(predictions, gold_entities)
            relaxed_results = self.evaluate_relaxed_matching(predictions, gold_entities)
            type_results = self.evaluate_by_entity_type(predictions, gold_entities)
            
            sample_results.append({
                'id': sample['id'],
                'strict': strict_results,
                'relaxed': relaxed_results,
                'by_type': type_results,
                'prediction_count': len(predictions),
                'gold_count': len(gold_entities)
            })
            
            # Collect for overall evaluation
            all_predictions.extend(predictions)
            all_gold.extend(gold_entities)
        
        # Overall evaluation
        overall_strict = self.evaluate_strict_matching(all_predictions, all_gold)
        overall_relaxed = self.evaluate_relaxed_matching(all_predictions, all_gold)
        overall_by_type = self.evaluate_by_entity_type(all_predictions, all_gold)
        
        print(f"Overall Results:")
        print(f"  Strict F1: {overall_strict['f1']:.3f}")
        print(f"  Relaxed F1: {overall_relaxed['f1']:.3f}")
        print(f"  Total predictions: {len(all_predictions)}")
        print(f"  Total gold entities: {len(all_gold)}")
        
        return {
            'overall': {
                'strict': overall_strict,
                'relaxed': overall_relaxed,
                'by_type': overall_by_type
            },
            'samples': sample_results,
            'threshold': threshold,
            'dataset_size': len(dataset)
        }
    
    def generate_confusion_matrix(self, predictions: List[Entity], gold_entities: List[Entity]) -> Dict:
        """Generate confusion matrix for entity types."""
        pred_types = [pred.entity_type for pred in predictions]
        gold_types = [gold.entity_type for gold in gold_entities]
        
        # Get all unique types
        all_types = sorted(set(pred_types + gold_types))
        
        # Create confusion matrix
        cm = confusion_matrix(gold_types, pred_types, labels=all_types)
        
        return {
            'matrix': cm,
            'labels': all_types,
            'pred_types': pred_types,
            'gold_types': gold_types
        }
    
    def plot_results(self, results: Dict, output_dir: str):
        """Generate plots for evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Plot overall performance
        metrics = ['precision', 'recall', 'f1']
        strict_values = [results['overall']['strict'][m] for m in metrics]
        relaxed_values = [results['overall']['relaxed'][m] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, strict_values, width, label='Strict Matching', alpha=0.8)
        bars2 = ax.bar(x + width/2, relaxed_values, width, label='Relaxed Matching', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('GLiNER Overall Performance on i2b2 2010')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot performance by entity type
        type_results = results['overall']['by_type']
        entity_types = list(type_results.keys())
        
        if entity_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Strict matching by type
            strict_f1 = [type_results[t]['strict']['f1'] for t in entity_types]
            bars1 = ax1.barh(entity_types, strict_f1)
            ax1.set_xlabel('F1 Score')
            ax1.set_title('Strict Matching F1 by Entity Type')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 1.0)
            
            # Add value labels
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                ax1.annotate(f'{width:.3f}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(3, 0),
                           textcoords="offset points",
                           ha='left', va='center')
            
            # Relaxed matching by type
            relaxed_f1 = [type_results[t]['relaxed']['f1'] for t in entity_types]
            bars2 = ax2.barh(entity_types, relaxed_f1)
            ax2.set_xlabel('F1 Score')
            ax2.set_title('Relaxed Matching F1 by Entity Type')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1.0)
            
            # Add value labels
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.annotate(f'{width:.3f}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(3, 0),
                           textcoords="offset points",
                           ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_by_type.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot entity type distribution
        if entity_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gold entity distribution
            gold_counts = [type_results[t]['gold_count'] for t in entity_types]
            ax1.pie(gold_counts, labels=entity_types, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Gold Entity Distribution')
            
            # Predicted entity distribution
            pred_counts = [type_results[t]['predicted_count'] for t in entity_types]
            ax2.pie(pred_counts, labels=entity_types, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Predicted Entity Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'entity_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_threshold_analysis(self, threshold_results: Dict, output_dir: str):
        """Plot threshold sensitivity analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        thresholds = sorted(threshold_results.keys())
        
        strict_f1 = [threshold_results[t]['strict']['f1'] for t in thresholds]
        strict_precision = [threshold_results[t]['strict']['precision'] for t in thresholds]
        strict_recall = [threshold_results[t]['strict']['recall'] for t in thresholds]
        
        relaxed_f1 = [threshold_results[t]['relaxed']['f1'] for t in thresholds]
        relaxed_precision = [threshold_results[t]['relaxed']['precision'] for t in thresholds]
        relaxed_recall = [threshold_results[t]['relaxed']['recall'] for t in thresholds]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Strict matching metrics
        ax1.plot(thresholds, strict_precision, 'o-', label='Precision', linewidth=2)
        ax1.plot(thresholds, strict_recall, 's-', label='Recall', linewidth=2)
        ax1.plot(thresholds, strict_f1, '^-', label='F1', linewidth=2)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Strict Matching - Threshold Sensitivity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Relaxed matching metrics
        ax2.plot(thresholds, relaxed_precision, 'o-', label='Precision', linewidth=2)
        ax2.plot(thresholds, relaxed_recall, 's-', label='Recall', linewidth=2)
        ax2.plot(thresholds, relaxed_f1, '^-', label='F1', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Relaxed Matching - Threshold Sensitivity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        # F1 comparison
        ax3.plot(thresholds, strict_f1, 'o-', label='Strict F1', linewidth=2)
        ax3.plot(thresholds, relaxed_f1, 's-', label='Relaxed F1', linewidth=2)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.0)
        
        # Entity count vs threshold
        entity_counts = [threshold_results[t]['strict']['true_positives'] + 
                        threshold_results[t]['strict']['false_positives'] for t in thresholds]
        ax4.plot(thresholds, entity_counts, 'o-', linewidth=2)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Number of Predicted Entities')
        ax4.set_title('Predicted Entity Count vs Threshold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Entity):
                return {
                    'start': obj.start,
                    'end': obj.end,
                    'text': obj.text,
                    'entity_type': obj.entity_type
                }
            return obj
        
        # Create serializable version
        serializable_results = json.loads(json.dumps(results, default=convert_types))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def generate_report(self, results: Dict, output_path: str):
        """Generate a detailed evaluation report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# GLiNER Evaluation Report - i2b2 2010 Dataset\n\n")
            
            # Overall results
            f.write("## Overall Performance\n\n")
            f.write("### Strict Matching\n")
            strict = results['overall']['strict']
            f.write(f"- **Precision**: {strict['precision']:.3f}\n")
            f.write(f"- **Recall**: {strict['recall']:.3f}\n")
            f.write(f"- **F1 Score**: {strict['f1']:.3f}\n")
            f.write(f"- **True Positives**: {strict['true_positives']}\n")
            f.write(f"- **False Positives**: {strict['false_positives']}\n")
            f.write(f"- **False Negatives**: {strict['false_negatives']}\n\n")
            
            f.write("### Relaxed Matching\n")
            relaxed = results['overall']['relaxed']
            f.write(f"- **Precision**: {relaxed['precision']:.3f}\n")
            f.write(f"- **Recall**: {relaxed['recall']:.3f}\n")
            f.write(f"- **F1 Score**: {relaxed['f1']:.3f}\n")
            f.write(f"- **True Positives**: {relaxed['true_positives']}\n")
            f.write(f"- **False Positives**: {relaxed['false_positives']}\n")
            f.write(f"- **False Negatives**: {relaxed['false_negatives']}\n\n")
            
            # Performance by entity type
            f.write("## Performance by Entity Type\n\n")
            type_results = results['overall']['by_type']
            
            f.write("| Entity Type | Strict F1 | Relaxed F1 | Gold Count | Pred Count |\n")
            f.write("|-------------|-----------|------------|------------|------------|\n")
            
            for entity_type, metrics in type_results.items():
                f.write(f"| {entity_type} | {metrics['strict']['f1']:.3f} | "
                       f"{metrics['relaxed']['f1']:.3f} | {metrics['gold_count']} | "
                       f"{metrics['predicted_count']} |\n")
            
            f.write("\n")
            
            # Dataset information
            f.write("## Dataset Information\n\n")
            f.write(f"- **Threshold**: {results['threshold']}\n")
            f.write(f"- **Dataset Size**: {results['dataset_size']} samples\n")
            f.write(f"- **Total Gold Entities**: {results['overall']['strict']['support']}\n")
            f.write(f"- **Total Predicted Entities**: {results['overall']['strict']['true_positives'] + results['overall']['strict']['false_positives']}\n")


def main():
    """Main evaluation function."""
    if not GLINER_AVAILABLE:
        print("GLiNER not available. Please install with: pip install gliner")
        return
    
    # Initialize evaluator
    evaluator = GLiNEREvaluator()
    
    # Load i2b2 2010 dataset
    print("Loading i2b2 2010 dataset...")
    i2b2_data_dir = "evaluation/data/i2b2_2010"

    i2b2_data_dir = os.path.abspath(i2b2_data_dir)
    print(f"Using i2b2 data directory: {i2b2_data_dir}")

    if not os.path.exists(i2b2_data_dir):
        print(f"Error: i2b2 data directory not found: {i2b2_data_dir}")
        print("Please download and extract the i2b2 2010 dataset to the data directory.")
        return
    
    try:
        i2b2_samples = evaluator.load_i2b2_2010_data(i2b2_data_dir)
        test_samples = [s for s in i2b2_samples if s['split'] == 'test']
        
        if not test_samples:
            print("No test samples found. Using all samples for evaluation.")
            test_samples = i2b2_samples
        
        print(f"Evaluating on {len(test_samples)} test samples")
        
        # Run evaluation
        results = evaluator.run_evaluation(test_samples)
        
        # Save results
        evaluator.save_results(results, "results/gliner_i2b2_results.json")
        
        # Generate plots
        evaluator.plot_results(results, "results/i2b2_plots")
        
        # Generate report
        evaluator.generate_report(results, "results/gliner_i2b2_report.md")
        
        # Run threshold analysis
        print("\nRunning threshold sensitivity analysis...")
        threshold_results = evaluator.analyze_threshold_sensitivity(test_samples)
        evaluator.plot_threshold_analysis(threshold_results, "results/i2b2_plots")
        
        print("\nEvaluation complete!")
        print(f"Results saved to: results/gliner_i2b2_results.json")
        print(f"Report saved to: results/gliner_i2b2_report.md")
        print(f"Plots saved to: results/i2b2_plots/")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()