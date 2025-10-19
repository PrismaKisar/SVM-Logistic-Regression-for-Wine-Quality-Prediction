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
            
            for t in range(1, self.n_iters + 1):
                # Random sample selection
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                # Compute g_t(x_t)
                g_t = sum(
                    alpha * kernel_function(sv, x_t, self.kernel, self.degree, self.gamma)
                    for alpha, sv in zip(self._alpha, self._support_vectors)
                )

                # Hinge loss: h_t(g_t) = max(0, 1 - y_t * g_t(x_t))
                hinge_loss = max(0, 1 - y_t * g_t)

                # Decay: g_t <- (1 - 1/t) * g_t
                self._alpha = [a * (1 - 1/t) for a in self._alpha]

                # Update only if hinge_loss > 0
                if hinge_loss > 0:
                    # Add new kernel: g_t <- g_t + (y_t / λt) * K(x_t, ·)
                    self._support_vectors.append(x_t.copy())
                    self._alpha.append(y_t / (self.lambda_param * t))
        
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
                g = sum(
                    alpha * kernel_function(sv, X[i], self.kernel, self.degree, self.gamma)
                    for alpha, sv in zip(self._alpha, self._support_vectors)
                )
                predictions[i] = np.sign(g) if g != 0 else 1

            return predictions


class LogisticRegression:
    # Threshold for adding support vectors: weight = σ(-y*g) must exceed this threshold
    _WEIGHT_THRESHOLD = 0.1

    def __init__(self, n_iters=1000, lambda_param=0.01, learning_rate=0.01, kernel='linear', degree=2, gamma=1.0, random_state=42):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state
        self._w = None
        self._b = None
        self._alpha = None
        self._X_train = None
        self._y_train = None

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if not np.array_equal(np.sort(np.unique(y)), np.array([-1, 1])):
            raise ValueError("y must contain only -1 and 1 values")

        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        if self.kernel == 'linear':
            self._w = np.zeros(n_features)
            self._b = 0.0

            for _ in range(self.n_iters):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                z_t = y_t * (np.dot(self._w, x_t) + self._b)

                sigma_term = self._logistic(-z_t)

                # Update w with regularization, b without
                self._w = (1 - self.learning_rate * self.lambda_param) * self._w + \
                        self.learning_rate * sigma_term * y_t * x_t

                self._b = self._b + self.learning_rate * sigma_term * y_t

        elif self.kernel in ['poly', 'gaussian']:
            self._alpha = []
            self._support_vectors = []
            
            for t in range(1, self.n_iters + 1):
                # Random sample selection
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                # Compute g_t(x_t)
                g_t = sum(
                    alpha * kernel_function(sv, x_t, self.kernel, self.degree, self.gamma)
                    for alpha, sv in zip(self._alpha, self._support_vectors)
                )

                # Continuous weight: σ(-y_t * g_t)
                weight = self._logistic(-y_t * g_t)

                # Decay: g_t <- (1 - 1/t) * g_t
                self._alpha = [a * (1 - 1/t) for a in self._alpha]

                # Add only if weight is significant (analogous to margin condition in SVM)
                # weight = σ(-y_t * g_t) is high when the model is uncertain or makes an error
                if weight > self._WEIGHT_THRESHOLD:
                    self._support_vectors.append(x_t.copy())
                    self._alpha.append((y_t / (self.lambda_param * t)) * weight)


    def _logistic(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        if self.kernel == 'linear':
            if self._w is None:
                raise ValueError("The model must be trained before any prediction")

            predictions = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                z = np.dot(self._w, X[i]) + self._b
                predictions[i] = self._logistic(z)
            return np.where(predictions >= 0.5, 1, -1)

        elif self.kernel in ['poly', 'gaussian']:
            if not self._alpha or not self._support_vectors:
                raise ValueError("The model must be trained before any prediction")

            predictions = np.array([
                self._logistic(sum(
                    alpha * kernel_function(sv, x, self.kernel, self.degree, self.gamma)
                    for alpha, sv in zip(self._alpha, self._support_vectors)
                ))
                for x in X
            ])

            return np.where(predictions >= 0.5, 1, -1)

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")