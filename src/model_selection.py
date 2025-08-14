import numpy as np

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
