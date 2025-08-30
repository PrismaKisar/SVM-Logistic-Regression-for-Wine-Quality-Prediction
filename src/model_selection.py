import numpy as np
from itertools import product
from joblib import Parallel, delayed
from util import calculate_metric, calculate_metrics

def _evaluate_params(model_class, params, X, y, cv, random_state):
    model = model_class(**params)
    all_scores = cross_val_score(model, X, y, cv=cv, random_state=random_state, metrics=['accuracy', 'precision', 'recall', 'f1'])
    return params, all_scores

def cross_val_score(model, X, y, cv=5, shuffle=True, random_state=42, metrics='accuracy'):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    fold_size = n_samples // cv
    remainder = n_samples % cv

    if isinstance(metrics, list):
        scores = {m: [] for m in metrics}
        start_idx = 0
        
        for i in range(cv):
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_fold_size
            
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            model.fit(X[train_indices], y[train_indices])
            predictions = model.predict(X[test_indices])
            y_test = y[test_indices]

            fold_metrics = calculate_metrics(predictions, y_test, metrics)
            
            for m in metrics:
                scores[m].append(fold_metrics[m])
            
            start_idx = end_idx
        
        return scores
    else:
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

            score = calculate_metric(predictions, y_test, metrics)
            scores.append(score)
            start_idx = end_idx
        
        return scores


def grid_search_cv(model_class, param_grid, X, y, cv=5, scoring='f1', random_state=42, n_jobs=-1):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    param_combinations = [dict(zip(param_names, combination)) for combination in product(*param_values)]
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_params)(model_class, params, X, y, cv, random_state)
        for params in param_combinations
    )
    
    best_score = 0
    best_params = None
    best_metrics = None
    
    for params, all_scores in results:
        mean_score = np.mean(all_scores[scoring])
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_metrics = {
                'accuracy': np.mean(all_scores['accuracy']),
                'precision': np.mean(all_scores['precision']),
                'recall': np.mean(all_scores['recall']),
                'f1': np.mean(all_scores['f1'])
            }
    
    return best_params, best_metrics