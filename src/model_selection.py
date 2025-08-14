import numpy as np
from itertools import product

def cross_val_score(model, X, y, cv=5, shuffle=True, random_state=42):
    pass
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

        accuracy = np.mean(predictions == y[test_indices])
        scores.append(accuracy)
        
        start_idx = end_idx
    
    return scores


def grid_search_cv(model_class, param_grid, X, y, cv=5):
    pass
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    best_score = 0
    best_params = None
    
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))
        model = model_class(**params)
        
        scores = cross_val_score(model, X, y, cv=cv)
        mean_score = np.mean(scores)
                
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, best_score
