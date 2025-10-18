import numpy as np

class SVM:
    def __init__(self, n_iters=1000, lambda_param=0.01, random_state=42, kernel='linear', degree=2, n_averaged_states=1):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.random_state = random_state
        self.kernel = kernel
        self.degree = degree
        self.n_averaged_states = n_averaged_states
        self._w = None
        self._b = None
        self._w_history = []
        self._b_history = []
        self._alpha = []
        self._support_vectors = []
        self._support_labels = []
        self._decision_history = []

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
        
        if not np.array_equal(np.unique(y), np.array([-1, 1])):
            raise ValueError("y must contain only -1 and 1 values")

        n_samples, n_features = X.shape
        np.random.seed(self.random_state)
        
        if self.kernel == 'linear':
            # Primal SVM using Pegasos algorithm
            # Keep weights and bias separate to avoid regularizing the bias
            self._w = np.zeros(n_features)
            self._b = 0.0
            self._w_history = []
            self._b_history = []

            for t in range(1, self.n_iters + 1):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                eta = 1 / (self.lambda_param * t)

                # Decision value: y*(w^T x + b). If > 0 correct, if < 1 inside margin
                decision_value = y_t * (np.dot(self._w, x_t) + self._b)

                # Gradient of: lambda*||w||^2 + max(0, 1 - y*(w^T*x + b))
                if decision_value < 1:
                    grad_w = self.lambda_param * self._w - y_t * x_t
                    grad_b = -y_t
                else:
                    grad_w = self.lambda_param * self._w
                    grad_b = 0.0

                self._w = self._w - eta * grad_w
                self._b = self._b - eta * grad_b

                self._w_history.append(self._w.copy())
                self._b_history.append(self._b)

            self._w = np.mean(self._w_history, axis=0)
            self._b = np.mean(self._b_history)
                
        elif self.kernel == 'poly':
            self._alpha = []
            self._support_vectors = []
            self._support_labels = []
            self._decision_history = []
            
            for t in range(1, self.n_iters + 1):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]
                
                decision = 0
                for alpha, y_sv, x_sv in zip(self._alpha, self._support_labels, self._support_vectors):
                    decision += alpha * y_sv * self._kernel_function(x_sv, x_t)
                
                h_t = max(0, 1 - y_t * decision)
                
                if h_t > 0:
                    self._alpha = [(1 - 1/t) * alpha for alpha in self._alpha]
                    self._alpha.append(1 / (self.lambda_param * t))
                    self._support_vectors.append(x_t.copy())
                    self._support_labels.append(y_t)
                else:
                    self._alpha = [(1 - 1/t) * alpha for alpha in self._alpha]
                
                threshold = 1e-6
                indices_to_keep = [i for i, alpha in enumerate(self._alpha) if alpha > threshold]
                
                self._alpha = [self._alpha[i] for i in indices_to_keep]
                self._support_vectors = [self._support_vectors[i] for i in indices_to_keep]
                self._support_labels = [self._support_labels[i] for i in indices_to_keep]
                
                step_interval = max(1, self.n_iters // self.n_averaged_states)
                if t % step_interval == 0:
                    current_state = {
                        'alpha': self._alpha.copy(),
                        'support_vectors': [sv.copy() for sv in self._support_vectors],
                        'support_labels': self._support_labels.copy()
                    }
                    self._decision_history.append(current_state)
        
    def predict(self, X):
        if self.kernel == 'linear':
            if self._w is None:
                raise ValueError("The model must be trained before any prediction")
            return np.sign(np.dot(X, self._w) + self._b)
            
        elif self.kernel == 'poly':
            if not self._decision_history:
                raise ValueError("The model must be trained before any prediction")
            
            predictions = np.zeros(X.shape[0])
            n_states = len(self._decision_history)
            
            for i in range(X.shape[0]):
                averaged_decision = 0
                x_i = X[i]
                
                for state in self._decision_history:
                    alpha_list = state['alpha']
                    sv_list = state['support_vectors']
                    
                    if alpha_list:
                        alphas = np.array(alpha_list)
                        svs = np.array(sv_list)
                        y_sv = np.array(state['support_labels'])
                        kernels = np.array([self._kernel_function(sv, x_i) for sv in svs])
                        decision = np.sum(alphas * y_sv * kernels)
                        averaged_decision += decision
                
                averaged_decision /= n_states
                predictions[i] = np.sign(averaged_decision) if averaged_decision != 0 else 1
            
            return predictions


class LogisticRegression:
    def __init__(self, n_iters=1000, lambda_param=0.01, learning_rate=0.01, kernel='linear', degree=2, random_state=42):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.degree = degree
        self.random_state = random_state
        self._w = None
        self._b = None

    def _expand_features(self, X):
        if self.kernel == 'linear':
            return X
        elif self.kernel == 'poly':
            from itertools import combinations_with_replacement
            
            n_samples, n_features = X.shape
            expanded_features = []
            
            for i in range(n_samples):
                x = X[i]
                expanded_x = [1]
                
                for degree in range(1, self.degree + 1):
                    for indices in combinations_with_replacement(range(n_features), degree):
                        term = 1
                        for idx in indices:
                            term *= x[idx]
                        expanded_x.append(term)
                
                expanded_features.append(expanded_x)
            
            return np.array(expanded_features)
        else:
            raise ValueError("The kernel must be one of 'linear' or 'poly'")

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if not np.array_equal(np.sort(np.unique(y)), np.array([-1, 1])):
            raise ValueError("y must contain only -1 and 1 values")

        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        if self.kernel == 'linear':
            # SGD for Logistic Regression with logistic loss
            # Keep weights and bias separate to avoid regularizing the bias
            self._w = np.zeros(n_features)
            self._b = 0.0

            for _ in range(self.n_iters):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                z_t = y_t * (np.dot(self._w, x_t) + self._b)
                # Gradient of: log(1 + e^(-y*z)) + (lambda/2)||w||^2
                sigma_term = self._logistic(-z_t)

                # Update w with regularization, b without
                self._w = (1 - self.learning_rate * self.lambda_param) * self._w + \
                        self.learning_rate * sigma_term * y_t * x_t

                self._b = self._b + self.learning_rate * sigma_term * y_t

        elif self.kernel == 'poly':
            # Polynomial kernel: use explicit feature expansion (includes bias in phi(x))
            X_expanded = self._expand_features(X)

            n_samples, n_features = X_expanded.shape
            self._w = np.zeros(n_features)

            for _ in range(self.n_iters):
                idx = np.random.randint(0, n_samples)
                x_t = X_expanded[idx]
                y_t = y[idx]

                z = np.dot(self._w, x_t)
                sigma_term = self._logistic(-y_t * z)
                gradient = -sigma_term * y_t * x_t + self.lambda_param * self._w
                self._w -= self.learning_rate * gradient

    def _logistic(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        if self._w is None:
            raise ValueError("The model must be trained before any prediction")

        if self.kernel == 'linear':
            predictions = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                z = np.dot(self._w, X[i]) + self._b
                predictions[i] = self._logistic(z)
            return np.where(predictions >= 0.5, 1, -1)

        elif self.kernel == 'poly':
            X_expanded = self._expand_features(X)
            predictions = np.zeros(X_expanded.shape[0])
            for i in range(X_expanded.shape[0]):
                z = np.dot(self._w, X_expanded[i])
                predictions[i] = self._logistic(z)
            return np.where(predictions >= 0.5, 1, -1)
    
