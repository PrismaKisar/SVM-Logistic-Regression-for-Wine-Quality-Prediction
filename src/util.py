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