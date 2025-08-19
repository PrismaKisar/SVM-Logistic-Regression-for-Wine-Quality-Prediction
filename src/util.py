import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df, correlation_threshold):
    correlation_matrix = df.corr(numeric_only=True)

    triangle_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    weak_corr_mask = abs(correlation_matrix) < correlation_threshold
    combined_mask = triangle_mask | weak_corr_mask

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix,
                mask=combined_mask,
                annot=True,
                cmap='coolwarm',
                center=0)

    plt.title(f'Strong Correlations (>{correlation_threshold})')
    plt.tight_layout()
    plt.show()


def split_train_test(X, y, test_size=0.2, random_state=42, stratify=None):

    np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)

    if stratify is not None:
        indices_train = []
        indices_test = []
        
        for class_val in np.unique(stratify):
            class_indices = np.where(stratify == class_val)[0]
            n_class_test = int(len(class_indices) * test_size)
            
            np.random.shuffle(class_indices)
            
            indices_test.extend(class_indices[:n_class_test])
            indices_train.extend(class_indices[n_class_test:])
        
        train_idx = np.array(indices_train)
        test_idx = np.array(indices_test)

        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)

    else:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    return X.iloc[train_idx].copy(), X.iloc[test_idx].copy(), y.iloc[train_idx].copy(), y.iloc[test_idx].copy()


def _confusion_matrix(predictions, y_true):
    tp = np.sum((predictions == 1) & (y_true == 1))
    fp = np.sum((predictions == 1) & (y_true == -1))
    tn = np.sum((predictions == -1) & (y_true == -1))
    fn = np.sum((predictions == -1) & (y_true == 1))
    return tp, fp, tn, fn


def calculate_accuracy(predictions, y_true):
    return np.mean(predictions == y_true)


def calculate_precision(predictions, y_true):
    tp, fp, _, _ = _confusion_matrix(predictions, y_true)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_recall(predictions, y_true):
    tp, _, _, fn = _confusion_matrix(predictions, y_true)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_f1(predictions, y_true):
    precision = calculate_precision(predictions, y_true)
    recall = calculate_recall(predictions, y_true)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def calculate_metric(predictions, y_true, metric):
    if metric == 'accuracy':
        return calculate_accuracy(predictions, y_true)
    elif metric == 'precision':
        return calculate_precision(predictions, y_true)
    elif metric == 'recall':
        return calculate_recall(predictions, y_true)
    elif metric == 'f1':
        return calculate_f1(predictions, y_true)
    else:
        raise ValueError("metric must be one of: accuracy, precision, recall, f1")


def calculate_metrics(predictions, y_true, metrics=['accuracy', 'precision', 'recall', 'f1']):
    if isinstance(metrics, str):
        metrics = [metrics]
    
    result = {}
    tp, fp, tn, fn = None, None, None, None
    
    for metric in metrics:
        if metric in ['precision', 'recall', 'f1'] and tp is None:
            tp, fp, tn, fn = _confusion_matrix(predictions, y_true)
        
        if metric == 'accuracy':
            result['accuracy'] = (tp + tn) / len(y_true) if tp is not None else np.mean(predictions == y_true)
        elif metric == 'precision':
            result['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric == 'recall':
            result['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == 'f1':
            precision = result.get('precision', tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recall = result.get('recall', tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            result['f1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    if tp is not None:
        result.update({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
    
    return result


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        X = X.values
        
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=1)
        
        return self
    
    def transform(self, X):        
        X_values = X.values
        columns = X.columns
        index = X.index
       
        X_scaled = (X_values - self.mean_) / self.scale_
        
        if columns is not None:
            return pd.DataFrame(X_scaled, columns=columns, index=index)
        else:
            return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)