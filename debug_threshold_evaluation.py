"""
Debug threshold evaluation to ensure calculations are correct
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch

def debug_threshold_evaluation():
    """Test threshold evaluation with a simple example."""
    
    # Create simple test case
    # 3 samples, 3 classes
    targets = np.array([
        [1, 0, 1],  # Sample 1: classes 0 and 2
        [0, 1, 0],  # Sample 2: class 1
        [1, 1, 0]   # Sample 3: classes 0 and 1
    ])
    
    # Predictions (probabilities)
    predictions = np.array([
        [0.8, 0.3, 0.7],  # High confidence for correct classes
        [0.2, 0.9, 0.1],  # High confidence for correct class
        [0.6, 0.4, 0.3]   # Medium confidence
    ])
    
    print("Test Data:")
    print("Targets:\n", targets)
    print("\nPredictions (probabilities):\n", predictions)
    
    # Test 1: Default threshold (0.5)
    print("\n" + "="*60)
    print("TEST 1: Default threshold = 0.5")
    pred_default = (predictions >= 0.5).astype(int)
    print("Binary predictions:\n", pred_default)
    
    # Manual calculation
    print("\nManual calculations:")
    for i in range(3):
        tp = ((pred_default[:, i] == 1) & (targets[:, i] == 1)).sum()
        fp = ((pred_default[:, i] == 1) & (targets[:, i] == 0)).sum()
        fn = ((pred_default[:, i] == 0) & (targets[:, i] == 1)).sum()
        tn = ((pred_default[:, i] == 0) & (targets[:, i] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_manual = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Sklearn calculation
        f1_sklearn = f1_score(targets[:, i], pred_default[:, i])
        
        print(f"Class {i}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision={precision:.3f}, Recall={recall:.3f}")
        print(f"  F1 (manual)={f1_manual:.3f}, F1 (sklearn)={f1_sklearn:.3f}")
        assert abs(f1_manual - f1_sklearn) < 0.001, "Manual and sklearn F1 don't match!"
    
    # Overall metrics
    macro_f1 = f1_score(targets, pred_default, average='macro')
    micro_f1 = f1_score(targets, pred_default, average='micro')
    print(f"\nOverall: Macro-F1={macro_f1:.3f}, Micro-F1={micro_f1:.3f}")
    
    # Test 2: Different thresholds per class
    print("\n" + "="*60)
    print("TEST 2: Per-class thresholds")
    thresholds = {0: 0.7, 1: 0.8, 2: 0.6}
    
    pred_optimal = np.zeros_like(predictions)
    for i in range(3):
        threshold = thresholds.get(i, 0.5)
        pred_optimal[:, i] = (predictions[:, i] >= threshold).astype(int)
        print(f"Class {i}: threshold={threshold}")
    
    print("\nBinary predictions with optimal thresholds:\n", pred_optimal)
    
    # Calculate metrics
    macro_f1_opt = f1_score(targets, pred_optimal, average='macro')
    micro_f1_opt = f1_score(targets, pred_optimal, average='micro')
    print(f"\nOptimal thresholds: Macro-F1={macro_f1_opt:.3f}, Micro-F1={micro_f1_opt:.3f}")
    
    # Compare
    print(f"\nDifference: Macro-F1 change = {macro_f1_opt - macro_f1:.3f}")
    
    # Test hamming loss calculation
    hamming_manual = np.mean(targets != pred_default)
    hamming_sklearn = 1 - f1_score(targets, pred_default, average='samples')
    print(f"\nHamming loss (manual): {hamming_manual:.3f}")
    
    # Test exact match ratio
    exact_match = np.mean(np.all(targets == pred_default, axis=1))
    print(f"Exact match ratio: {exact_match:.3f}")
    
    print("\n" + "="*60)
    print("All calculations verified!")


def check_threshold_optimization_issue():
    """Check why threshold optimization might degrade performance."""
    
    print("\nPOSSIBLE ISSUES WITH THRESHOLD OPTIMIZATION:")
    print("="*60)
    
    print("\n1. Validation vs Test Distribution:")
    print("   - If validation set has different class distribution than test set")
    print("   - Thresholds optimized on validation won't generalize well")
    
    print("\n2. Optimization Metric:")
    print("   - Optimizing per-class F1 doesn't guarantee better macro/micro F1")
    print("   - Example: if threshold makes one class much worse, overall metric suffers")
    
    print("\n3. Threshold Search Resolution:")
    print("   - precision_recall_curve might miss the truly optimal threshold")
    print("   - Grid search with finer resolution might help")
    
    print("\n4. Class Imbalance:")
    print("   - Rare classes might get extreme thresholds (close to 0 or 1)")
    print("   - These don't generalize well")
    
    # Simulate a case where per-class optimization hurts overall performance
    print("\n" + "="*60)
    print("EXAMPLE: When per-class optimization hurts overall metrics")
    
    # Create imbalanced example
    np.random.seed(42)
    n_samples = 100
    
    # Class 0: common, well-separated
    # Class 1: rare, overlapping
    targets = np.zeros((n_samples, 2))
    predictions = np.zeros((n_samples, 2))
    
    # 80% have class 0
    targets[:80, 0] = 1
    predictions[:80, 0] = np.random.uniform(0.6, 0.9, 80)  # High confidence
    predictions[80:, 0] = np.random.uniform(0.1, 0.4, 20)  # Low confidence
    
    # 10% have class 1
    targets[40:50, 1] = 1
    predictions[40:50, 1] = np.random.uniform(0.4, 0.6, 10)  # Medium confidence
    predictions[:40, 1] = np.random.uniform(0.3, 0.5, 40)    # Close to threshold
    predictions[50:, 1] = np.random.uniform(0.3, 0.5, 50)    # Close to threshold
    
    # Default threshold
    pred_default = (predictions >= 0.5).astype(int)
    macro_f1_default = f1_score(targets, pred_default, average='macro')
    
    # Find "optimal" thresholds
    from sklearn.metrics import precision_recall_curve
    
    optimal_thresholds = {}
    for i in range(2):
        precisions, recalls, thresholds = precision_recall_curve(targets[:, i], predictions[:, i])
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_thresholds[i] = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Apply optimal thresholds
    pred_optimal = np.zeros_like(predictions)
    for i in range(2):
        pred_optimal[:, i] = (predictions[:, i] >= optimal_thresholds[i]).astype(int)
    
    macro_f1_optimal = f1_score(targets, pred_optimal, average='macro')
    
    print(f"\nClass distribution: Class 0: {targets[:, 0].sum()}, Class 1: {targets[:, 1].sum()}")
    print(f"Optimal thresholds: Class 0: {optimal_thresholds[0]:.3f}, Class 1: {optimal_thresholds[1]:.3f}")
    print(f"Default threshold (0.5): Macro-F1 = {macro_f1_default:.3f}")
    print(f"Optimal thresholds: Macro-F1 = {macro_f1_optimal:.3f}")
    print(f"Change: {macro_f1_optimal - macro_f1_default:.3f}")
    
    if macro_f1_optimal < macro_f1_default:
        print("\n⚠️  Optimal thresholds performed WORSE than default!")
        print("This can happen when:")
        print("- Threshold optimization overfits to validation set")
        print("- Per-class optimization doesn't align with overall metric")
        print("- Class imbalance leads to extreme thresholds")


if __name__ == "__main__":
    debug_threshold_evaluation()
    check_threshold_optimization_issue()