import numpy as np

class SVM:
    def __init__(self, n_iters=1000, lambda_param=0.01, random_seed=42):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        self.w = None
    
    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if not np.array_equal(np.sort(np.unique(y)), np.array([-1, 1])):
            raise ValueError("y must contain only -1 and 1 values")

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
        if self.w is None:
            raise ValueError("The model must be trained before any prediction")
        
        return np.sign(np.dot(X, self.w))
    
class LogisticRegression:
    def __init__(self, n_iters=1000, lambda_param=0.01, learning_rate=0.01):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.w = None

    def fit(self, X, y):
        bias_column = np.ones((X.shape[0], 1))
        X = np.hstack([bias_column, X])
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)

        for _ in range(self.n_iters):
            for t in range(n_samples):
                x_t = X[t]
                y_t = y[t]

                gradient = -self.logistic(-y_t * np.dot(self.w, x_t)) * y_t * x_t + self.lambda_param * self.w
                self.w -= self.learning_rate * gradient

    def logistic(self, z):
        return 1 / (1 + np.e**-z)

    def predict(self, X):
        bias_column = np.ones((X.shape[0], 1))
        X = np.hstack([bias_column, X])
    
        y = self.logistic(np.dot(X, self.w))
        return np.where(y >= 0.5, 1, -1)
    
