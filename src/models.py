import numpy as np

class SVM:
    def __init__(self, n_iters=1000, lambda_param=0.01, random_seed=42):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        self.w = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        
        rng = np.random.RandomState(self.random_seed)
        
        for t in range(1, self.n_iters + 1):
            idx = rng.randint(0, n_samples)
            x_t = X[idx]
            y_t = y[idx]
            
            eta = 1 / (self.lambda_param * t)
            margin = y_t * np.dot(self.w, x_t)

            if margin < 1:
                gradient = self.lambda_param * self.w - y_t * x_t
            else:
                gradient = self.lambda_param * self.w

            self.w = self.w - eta * gradient
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w))
    