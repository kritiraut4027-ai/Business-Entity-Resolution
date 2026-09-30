import numpy as np

def compute_entity_metrics(pred_set, true_set):
    """
    Computes precision, recall, and F0.5 for a single Source 1 entity according to official challenge spec.
    """
    # Case 1: Singleton (No true matches)
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0 # Perfect prediction on singleton
        else:
            return 0.0, 1.0, 0.0 # False merge on singleton penalizes to 0.0

    # Case 2: Entity has true matches, but model predicted none
    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    # Case 3: Entity has true matches and predictions made
    tp = len(pred_set.intersection(true_set))
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    denominator = (0.25 * precision) + recall
    if denominator == 0 or (precision == 0 and recall == 0):
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denominator
        
    return precision, recall, f05


def evaluate_predictions(predictions_dict, ground_truth_dict):
    """
    Computes macro-averaged Precision, Recall, and F0.5 across all Source 1 entities in evaluation set.
    predictions_dict: {s1_id: set_of_predicted_ids}
    ground_truth_dict: {s1_id: set_of_true_ids}
    """
    all_s1_ids = set(ground_truth_dict.keys())
    
    precisions = []
    recalls = []
    f05_scores = []
    
    singleton_count = 0
    correct_singletons = 0
    
    for s1_id in all_s1_ids:
        true_set = ground_truth_dict[s1_id]
        pred_set = predictions_dict.get(s1_id, set())
        
        prec, rec, f05 = compute_entity_metrics(pred_set, true_set)
        precisions.append(prec)
        recalls.append(rec)
        f05_scores.append(f05)
        
        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                correct_singletons += 1
                
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f05 = float(np.mean(f05_scores))
    singleton_acc = (correct_singletons / singleton_count) if singleton_count > 0 else 1.0
    
    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "singleton_accuracy": singleton_acc,
        "total_evaluated_entities": len(all_s1_ids)
    }
