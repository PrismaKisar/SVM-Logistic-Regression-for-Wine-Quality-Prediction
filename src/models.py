import numpy as np

class SVM:
    def __init__(self, n_iters=1000, lambda_param=0.01, random_seed=42, kernel='linear', degree=2):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        self.kernel = kernel
        self.degree = degree
        self._w = None
        self._S = None
        self._X_errors = None
        self._y_errors = None
    
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (1 + np.dot(x1, x2))**self.degree
        else:
            raise ValueError("The kernel must be one of 'linear' or 'poly'")
    
    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if not np.array_equal(np.sort(np.unique(y)), np.array([-1, 1])):
            raise ValueError("y must contain only -1 and 1 values")

        n_samples, n_features = X.shape
        np.random.seed(self.random_seed)
        
        if self.kernel == 'linear':
            self._w = np.zeros(n_features)
            
            for t in range(1, self.n_iters + 1):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]
                
                eta = 1 / (self.lambda_param * t)
                margin = y_t * self._kernel_function(self._w, x_t)

                if margin < 1:
                    gradient = self.lambda_param * self._w - y_t * x_t
                else:
                    gradient = self.lambda_param * self._w

                self._w = self._w - eta * gradient
                
        elif self.kernel == 'poly':
            self._S = []
            self._X_errors = []
            self._y_errors = []
            
            for t in range(self.n_iters):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]
                
                y_hat = 0
                for i, s in enumerate(self._S):
                    y_hat += self._y_errors[i] * self._kernel_function(self._X_errors[i], x_t)
                
                y_hat = np.sign(y_hat) if y_hat != 0 else 1
                
                if y_hat != y_t:
                    self._S.append(t)
                    self._X_errors.append(x_t.copy())
                    self._y_errors.append(y_t)
        else:
            raise ValueError("The kernel must be one of 'linear' or 'poly'")
    
    def predict(self, X):
        if self.kernel == 'linear':
            if self._w is None:
                raise ValueError("The model must be trained before any prediction")
            
            predictions = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                predictions[i] = np.sign(self._kernel_function(self._w, X[i]))
            return predictions
            
        elif self.kernel == 'poly':
            if self._S is None:
                raise ValueError("The model must be trained before any prediction")
            
            n_test = X.shape[0]
            predictions = np.zeros(n_test)
            
            for i in range(n_test):
                decision_value = 0
                for j in range(len(self._S)):
                    decision_value += self._y_errors[j] * self._kernel_function(self._X_errors[j], X[i])
                
                predictions[i] = np.sign(decision_value) if decision_value != 0 else 1
            
            return predictions


class LogisticRegression:
    def __init__(self, n_iters=1000, lambda_param=0.01, learning_rate=0.01):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self._w = None

    def _expand_features(self, X):
        if self.kernel == 'linear':
            return X
        elif self.kernel == 'poly':
            n_samples = X.shape[0]
            n_features = X.shape[1]
            
            if self.degree == 2:
                expanded_features = []
                for i in range(n_samples):
                    x = X[i]
                    expanded_x = [1]
                    
                    for j in range(n_features):
                        expanded_x.append(x[j])
                    
                    for j in range(n_features):
                        for k in range(j, n_features):
                            expanded_x.append(x[j] * x[k])
                    
                    expanded_features.append(expanded_x)
                
                return np.array(expanded_features)
            
            elif self.degree == 3:
                expanded_features = []
                for i in range(n_samples):
                    x = X[i]
                    expanded_x = [1]
                    
                    for j in range(n_features):
                        expanded_x.append(x[j])
                    
                    for j in range(n_features):
                        for k in range(j, n_features):
                            expanded_x.append(x[j] * x[k])
                    
                    for j in range(n_features):
                        for k in range(j, n_features):
                            for l in range(k, n_features):
                                expanded_x.append(x[j] * x[k] * x[l])
                    
                    expanded_features.append(expanded_x)
                
                return np.array(expanded_features)
            
            else:
                raise ValueError("Only degree 2 and 3 are supported for polynomial expansion")
        else:
            raise ValueError("The kernel must be one of 'linear' or 'poly'")

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if not np.array_equal(np.sort(np.unique(y)), np.array([-1, 1])):
            raise ValueError("y must contain only -1 and 1 values")

        X_expanded = self._expand_features(X)
        
        if self.kernel == 'linear':
            bias_column = np.ones((X_expanded.shape[0], 1))
            X_expanded = np.hstack([bias_column, X_expanded])
        
        n_samples, n_features = X_expanded.shape
        self._w = np.zeros(n_features)

        for _ in range(self.n_iters):
            for t in range(n_samples):
                x_t = X_expanded[t]
                y_t = y[t]

                z = np.dot(self._w, x_t)
                gradient = -self._logistic(-y_t * z) * y_t * x_t + self.lambda_param * self._w
                self._w -= self.learning_rate * gradient

    def _logistic(self, z):
        return 1 / (1 + np.e**-z)

    def predict(self, X):
        if self._w is None:
            raise ValueError("The model must be trained before any prediction")

        bias_column = np.ones((X.shape[0], 1))
        X = np.hstack([bias_column, X])
    
        y = self._logistic(np.dot(X, self._w))
        return np.where(y >= 0.5, 1, -1)
    
