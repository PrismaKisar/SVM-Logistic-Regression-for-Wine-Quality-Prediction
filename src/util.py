import numpy as np
import pandas as pd

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