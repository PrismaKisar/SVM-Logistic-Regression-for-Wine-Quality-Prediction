import numpy as np
from itertools import product

def cross_val_score(model, X, y, cv=5, shuffle=True, random_state=42, metric='accuracy'):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    fold_size = n_samples // cv
    remainder = n_samples % cv

    scores = []
    start_idx = 0
    
    for i in range(cv):
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_fold_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        model.fit(X[train_indices], y[train_indices])
        predictions = model.predict(X[test_indices])
        y_test = y[test_indices]

        if metric == 'accuracy':
            score = np.mean(predictions == y_test)
        elif metric == 'precision':
            tp = np.sum((predictions == 1) & (y_test == 1))
            fp = np.sum((predictions == 1) & (y_test == 0))
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric == 'recall':
            tp = np.sum((predictions == 1) & (y_test == 1))
            fn = np.sum((predictions == 0) & (y_test == 1))
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == 'f1':
            tp = np.sum((predictions == 1) & (y_test == 1))
            fp = np.sum((predictions == 1) & (y_test == 0))
            fn = np.sum((predictions == 0) & (y_test == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            raise ValueError("metric must be one of: accuracy, precision, recall, f1")
        
        scores.append(score)
        start_idx = end_idx
    
    return scores


def grid_search_cv(model_class, param_grid, X, y, cv=5):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    best_f1 = 0
    best_params = None
    best_metrics = None
    
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))
        model = model_class(**params)
        
        precision_scores = cross_val_score(model, X, y, cv=cv, metric='precision')
        recall_scores = cross_val_score(model, X, y, cv=cv, metric='recall')
        f1_scores = cross_val_score(model, X, y, cv=cv, metric='f1')
                
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_f1 = np.mean(f1_scores)
        
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params
            best_metrics = {
                'precision': mean_precision,
                'recall': mean_recall,
                'f1': mean_f1
            }
    
    return best_params, best_metrics


