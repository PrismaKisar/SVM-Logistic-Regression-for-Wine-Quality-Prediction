import numpy as np


def kernel_function(x1, x2, kernel='poly', degree=2, gamma=1.0):
    if kernel == 'poly':
        return (1 + np.dot(x1, x2))**degree
    elif kernel == 'gaussian':
        return np.exp(-1 / (2 * gamma) * np.linalg.norm(x1 - x2)**2)
    else:
        raise ValueError("The kernel must be one of 'poly' or 'gaussian'")



class SVM:
    def __init__(self, n_iters=1000, lambda_param=0.01, random_state=42, kernel='linear', degree=2, gamma=1.0):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.random_state = random_state
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self._w = None
        self._b = None
        self._alpha = []
        self._support_vectors = []
        self._support_labels = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        if self.kernel == 'linear':
            self._w = np.zeros(n_features)
            self._b = 0.0

            for t in range(1, self.n_iters + 1):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                eta = 1 / (self.lambda_param * t)
                decision_value = y_t * (np.dot(self._w, x_t) + self._b)

                if decision_value < 1:
                    self._w = (1 - 1/t) * self._w + eta * y_t * x_t
                    self._b += eta * y_t
                else:
                    self._w = (1 - 1/t) * self._w

        elif self.kernel in ['poly', 'gaussian']:
            self._alpha = []
            self._support_vectors = []
            self._support_labels = []

            for t in range(1, self.n_iters + 1):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                # Decision function f(x_t)
                f_x = sum(
                    alpha * kernel_function(sv, x_t, self.kernel, self.degree, self.gamma)
                    for alpha, sv in zip(self._alpha, self._support_vectors)
                )

                # Decision value y_t * g_t(x_t)
                decision_value = y_t * f_x

                # Decadimento g_t <- (1 - 1/t) g_t
                self._alpha = [a * (1 - 1/t) for a in self._alpha]

                # Aggiornamento solo se dentro il margine
                if decision_value < 1:
                    self._support_vectors.append(x_t.copy())
                    self._support_labels.append(y_t)
                    self._alpha.append(y_t / (self.lambda_param * t))

            # Pulisci i termini con alpha troppo piccoli
            threshold = 1e-6
            filtered = [(sv, y_sv, a) for sv, y_sv, a in zip(
                self._support_vectors, self._support_labels, self._alpha
            ) if abs(a) > threshold]

            if filtered:
                self._support_vectors, self._support_labels, self._alpha = zip(*filtered)
                self._support_vectors = list(self._support_vectors)
                self._support_labels = list(self._support_labels)
                self._alpha = list(self._alpha)
        
    def predict(self, X):
        if self.kernel == 'linear':
            if self._w is None:
                raise ValueError("The model must be trained before any prediction")
            return np.sign(np.dot(X, self._w) + self._b)

        elif self.kernel in ['poly', 'gaussian']:
            if not self._support_vectors:
                raise ValueError("The model must be trained before any prediction")

            predictions = np.zeros(X.shape[0])

            for i in range(X.shape[0]):
                f_x = sum(
                    alpha * kernel_function(sv, X[i], self.kernel, self.degree, self.gamma)
                    for alpha, sv in zip(self._alpha, self._support_vectors)
                )
                predictions[i] = np.sign(f_x) if f_x != 0 else 1

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
    
