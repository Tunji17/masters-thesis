"""
Evaluation metrics for relation extraction tasks.
Implements precision, recall, F1 score, and triple-level accuracy for scientific text and entities relations.
"""

from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import numpy as np
# from sklearn.metrics import precision_recall_fscore_support, classification_report


class RelationMetrics:
    """Evaluation metrics for relation extraction."""

    def __init__(self):
        self.reset()
        # Define symmetric relation types
        self.symmetric_relations = {'Association', 'Bind', 'Comparison'}

    def reset(self):
        """Reset all metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.predictions = []
        self.ground_truth = []
    
    def normalize_triple(self, triple: Tuple[str, str, str, str]) -> Tuple[str, str, str, str]:
        """Normalize a triple for comparison."""
        pmid, ent1, ent2, rel = triple
        return (pmid, ent1.lower().strip(), ent2.lower().strip(), rel)
    
    def _expand_with_bidirectional(self, triples: List[Tuple[str, str, str, str]]) -> Set[Tuple[str, str, str, str]]:
        """Expand triples to include bidirectional relations for symmetric types."""
        expanded = set(triples)
        for pmid, ent1, ent2, rel in triples:
            if rel in self.symmetric_relations:
                # Add reversed triple for symmetric relations
                expanded.add((pmid, ent2, ent1, rel))
        return expanded
    
    def _entities_match_partial(self, ent1: str, ent2: str) -> bool:
        """Check if entities match partially (substring matching)."""
        ent1_lower = ent1.lower().strip()
        ent2_lower = ent2.lower().strip()
        
        # Check if one is substring of the other
        return ent1_lower in ent2_lower or ent2_lower in ent1_lower
    
    def _prepare_partial_matching(self, pred_triples: List[Tuple[str, str, str, str]], 
                                  gt_triples: List[Tuple[str, str, str, str]]) -> Tuple[Set, Set]:
        """Prepare sets for partial matching by finding matches with substring entity matching."""
        matched_pred = set()
        matched_gt = set()
        unmatched_pred = set(pred_triples)
        unmatched_gt = set(gt_triples)
        
        # First pass: exact matches
        exact_matches = set(pred_triples) & set(gt_triples)
        matched_pred.update(exact_matches)
        matched_gt.update(exact_matches)
        unmatched_pred -= exact_matches
        unmatched_gt -= exact_matches
        
        # Second pass: partial entity matches
        for pred in list(unmatched_pred):
            pmid_p, ent1_p, ent2_p, rel_p = pred
            for gt in list(unmatched_gt):
                pmid_g, ent1_g, ent2_g, rel_g = gt
                
                # Check if pmid and relation match
                if pmid_p == pmid_g and rel_p == rel_g:
                    # Check if entities match (considering both orders)
                    if ((self._entities_match_partial(ent1_p, ent1_g) and 
                         self._entities_match_partial(ent2_p, ent2_g)) or
                        (self._entities_match_partial(ent1_p, ent2_g) and 
                         self._entities_match_partial(ent2_p, ent1_g))):
                        # Found a partial match
                        matched_pred.add(pred)
                        matched_gt.add(gt)
                        unmatched_pred.remove(pred)
                        unmatched_gt.remove(gt)
                        break
        
        # Return the matched sets (which will be used for intersection)
        # and unmatched items will be treated as FP/FN
        return matched_pred | unmatched_pred, matched_gt | unmatched_gt

    def add_predictions(self, predicted_triples: List[Tuple[str, str, str, str]],
                        ground_truth_triples: List[Tuple[str, str, str, str]],
                        matching_mode: str = 'strict',
                        handle_bidirectional: bool = False):
        """
        Add predictions and ground truth for evaluation.

        Args:
            predicted_triples: List of (pmid, chemical, disease, relation) tuples
            ground_truth_triples: List of (pmid, chemical, disease, relation) tuples
            matching_mode: 'strict' (exact match), 'normalized' (case-insensitive), or 'partial' (substring)
            handle_bidirectional: Whether to handle bidirectional relations for symmetric types
        """
        self.predictions.extend(predicted_triples)
        self.ground_truth.extend(ground_truth_triples)

        # Expand with bidirectional relations if requested
        if handle_bidirectional:
            pred_expanded = self._expand_with_bidirectional(predicted_triples)
            gt_expanded = self._expand_with_bidirectional(ground_truth_triples)
        else:
            pred_expanded = set(predicted_triples)
            gt_expanded = set(ground_truth_triples)

        if matching_mode == 'strict':
            # Use expanded sets directly
            pred_set = pred_expanded
            gt_set = gt_expanded
        elif matching_mode == 'normalized':
            # Normalize after expansion
            pred_set = {self.normalize_triple(t) for t in pred_expanded}
            gt_set = {self.normalize_triple(t) for t in gt_expanded}
        elif matching_mode == 'partial':
            # For partial matching, we'll need custom logic
            pred_set, gt_set = self._prepare_partial_matching(list(pred_expanded), list(gt_expanded))
        else:
            raise ValueError(f"Unknown matching_mode: {matching_mode}")

        # print(f"Pred_set: {pred_set}")
        # print(f"GT_set: {gt_set}")
        # Calculate TP, FP, FN
        self.true_positives += len(pred_set & gt_set)
        self.false_positives += len(pred_set - gt_set)
        self.false_negatives += len(gt_set - pred_set)

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate precision, recall, F1 score."""
        if self.true_positives + self.false_positives == 0:
            precision = 0.0
        else:
            precision = self.true_positives / (self.true_positives + self.false_positives)

        if self.true_positives + self.false_negatives == 0:
            recall = 0.0
        else:
            recall = self.true_positives / (self.true_positives + self.false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }

    def calculate_document_level_metrics(self) -> Dict[str, float]:
        """Calculate metrics at document level."""
        # Group by PMID
        pred_by_pmid = defaultdict(set)
        gt_by_pmid = defaultdict(set)

        for triple in self.predictions:
            pmid = triple[0]
            pred_by_pmid[pmid].add(triple[1:])  # (chemical, disease, relation)

        for triple in self.ground_truth:
            pmid = triple[0]
            gt_by_pmid[pmid].add(triple[1:])  # (chemical, disease, relation)

        # Calculate metrics per document
        doc_precisions = []
        doc_recalls = []
        doc_f1s = []

        all_pmids = set(pred_by_pmid.keys()) | set(gt_by_pmid.keys())

        for pmid in all_pmids:
            pred_rels = pred_by_pmid[pmid]
            gt_rels = gt_by_pmid[pmid]

            if len(pred_rels) == 0:
                precision = 0.0
            else:
                precision = len(pred_rels & gt_rels) / len(pred_rels)

            if len(gt_rels) == 0:
                recall = 0.0
            else:
                recall = len(pred_rels & gt_rels) / len(gt_rels)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            doc_precisions.append(precision)
            doc_recalls.append(recall)
            doc_f1s.append(f1)

        return {
            'doc_avg_precision': np.mean(doc_precisions),
            'doc_avg_recall': np.mean(doc_recalls),
            'doc_avg_f1': np.mean(doc_f1s),
            'doc_std_precision': np.std(doc_precisions),
            'doc_std_recall': np.std(doc_recalls),
            'doc_std_f1': np.std(doc_f1s)
        }

    def get_error_analysis(self) -> Dict[str, List[Tuple]]:
        """Get detailed error analysis."""
        pred_set = set(self.predictions)
        gt_set = set(self.ground_truth)

        return {
            'false_positives': list(pred_set - gt_set),
            'false_negatives': list(gt_set - pred_set),
            'true_positives': list(pred_set & gt_set)
        }
    
    def get_near_matches(self) -> Dict[str, List[Tuple]]:
        """Identify near matches - same entities but different relations."""
        near_matches = []
        entity_only_matches = []
        
        pred_set = set(self.predictions)
        gt_set = set(self.ground_truth)
        
        # Get false positives and false negatives
        false_positives = pred_set - gt_set
        false_negatives = gt_set - pred_set
        
        # Check for near matches
        for fp in false_positives:
            pmid_fp, ent1_fp, ent2_fp, rel_fp = fp
            for fn in false_negatives:
                pmid_fn, ent1_fn, ent2_fn, rel_fn = fn
                
                # Same document check
                if pmid_fp == pmid_fn:
                    # Check if entities match (exact or reversed)
                    if ((ent1_fp == ent1_fn and ent2_fp == ent2_fn) or 
                        (ent1_fp == ent2_fn and ent2_fp == ent1_fn)):
                        # Same entities, different relation
                        near_matches.append({
                            'predicted': fp,
                            'ground_truth': fn,
                            'mismatch_type': 'relation_only'
                        })
                    # Check for partial entity matches
                    elif (self._entities_match_partial(ent1_fp, ent1_fn) or 
                          self._entities_match_partial(ent1_fp, ent2_fn) or
                          self._entities_match_partial(ent2_fp, ent1_fn) or
                          self._entities_match_partial(ent2_fp, ent2_fn)):
                        entity_only_matches.append({
                            'predicted': fp,
                            'ground_truth': fn,
                            'mismatch_type': 'entity_partial'
                        })
        
        return {
            'near_matches': near_matches,
            'entity_partial_matches': entity_only_matches,
            'total_near_matches': len(near_matches) + len(entity_only_matches)
        }
    
    def calculate_relaxed_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics under different matching strategies."""
        # Store current state
        original_tp = self.true_positives
        original_fp = self.false_positives
        original_fn = self.false_negatives
        
        results = {}
        
        # Test different matching modes
        modes = [
            ('strict', False),
            ('normalized', False),
            ('normalized_bidirectional', True),
            ('partial', False),
            ('partial_bidirectional', True)
        ]
        
        for mode_name, (mode, bidirectional) in [('strict', ('strict', False)), 
                                                  ('normalized', ('normalized', False)),
                                                  ('normalized_bidirectional', ('normalized', True)),
                                                  ('partial', ('partial', False)),
                                                  ('partial_bidirectional', ('partial', True))]:
            # Reset metrics
            self.true_positives = 0
            self.false_positives = 0
            self.false_negatives = 0
            
            # Re-calculate with different mode
            if mode == 'strict':
                pred_set = set(self.predictions)
                gt_set = set(self.ground_truth)
            else:
                # Create temporary expanded sets if bidirectional
                if bidirectional:
                    pred_expanded = self._expand_with_bidirectional(self.predictions)
                    gt_expanded = self._expand_with_bidirectional(self.ground_truth)
                else:
                    pred_expanded = set(self.predictions)
                    gt_expanded = set(self.ground_truth)
                
                if mode == 'normalized':
                    pred_set = {self.normalize_triple(t) for t in pred_expanded}
                    gt_set = {self.normalize_triple(t) for t in gt_expanded}
                elif mode == 'partial':
                    pred_set, gt_set = self._prepare_partial_matching(list(pred_expanded), list(gt_expanded))
            
            # Calculate metrics
            self.true_positives = len(pred_set & gt_set)
            self.false_positives = len(pred_set - gt_set)
            self.false_negatives = len(gt_set - pred_set)
            
            metrics = self.calculate_metrics()
            results[mode_name] = metrics
        
        # Restore original state
        self.true_positives = original_tp
        self.false_positives = original_fp
        self.false_negatives = original_fn
        
        return results

    def print_evaluation_report(self):
        """Print detailed evaluation report."""
        metrics = self.calculate_metrics()
        doc_metrics = self.calculate_document_level_metrics()

        print("=== Relation Extraction Evaluation Report ===")
        print(f"Total Predictions: {len(self.predictions)}")
        print(f"Total Ground Truth: {len(self.ground_truth)}")
        print()

        print("Triple-Level Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print()

        print("Document-Level Metrics:")
        print(f"  Avg Precision: {doc_metrics['doc_avg_precision']:.4f} (±{doc_metrics['doc_std_precision']:.4f})")
        print(f"  Avg Recall: {doc_metrics['doc_avg_recall']:.4f} (±{doc_metrics['doc_std_recall']:.4f})")
        print(f"  Avg F1 Score: {doc_metrics['doc_avg_f1']:.4f} (±{doc_metrics['doc_std_f1']:.4f})")
        print()

        # Error analysis
        errors = self.get_error_analysis()
        print("Error Analysis:")
        print(f"  False Positives: {len(errors['false_positives'])}")
        print(f"  False Negatives: {len(errors['false_negatives'])}")
        print(f"  True Positives: {len(errors['true_positives'])}")

        if errors['false_positives']:
            print("\nSample False Positives:")
            for i, fp in enumerate(errors['false_positives'][:5]):
                print(f"  {i+1}. {fp}")

        if errors['false_negatives']:
            print("\nSample False Negatives:")
            for i, fn in enumerate(errors['false_negatives'][:5]):
                print(f"  {i+1}. {fn}")
        
        # Near match analysis
        near_match_info = self.get_near_matches()
        if near_match_info['total_near_matches'] > 0:
            print(f"\nNear Matches Analysis:")
            print(f"  Relation-only mismatches: {len(near_match_info['near_matches'])}")
            print(f"  Entity partial matches: {len(near_match_info['entity_partial_matches'])}")
            
            if near_match_info['near_matches']:
                print("\nSample Relation-only Mismatches:")
                for i, match in enumerate(near_match_info['near_matches'][:3]):
                    print(f"  {i+1}. Predicted: {match['predicted']}")
                    print(f"     Expected: {match['ground_truth']}")
        
        # Relaxed metrics
        print("\nRelaxed Metrics Comparison:")
        relaxed = self.calculate_relaxed_metrics()
        for mode, metrics in relaxed.items():
            print(f"  {mode}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, TP={metrics['true_positives']}")
