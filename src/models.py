import numpy as np


def kernel_function(x1, x2, kernel='poly', degree=2, gamma=1.0):
    if x1.ndim == 1 and x2.ndim == 1:
        if kernel == 'poly':
            return (1 + np.dot(x1, x2))**degree
        elif kernel == 'gaussian':
            return np.exp(-1 / (2 * gamma) * np.linalg.norm(x1 - x2)**2)
        else:
            raise ValueError("The kernel must be one of 'poly' or 'gaussian'")


    elif x1.ndim == 2 and x2.ndim == 2:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                if kernel == 'poly':
                    K[i, j] = (1 + np.dot(x1[i], x2[j]))**degree
                elif kernel == 'gaussian':
                    K[i, j] = np.exp(-1 / (2 * gamma) * np.linalg.norm(x1[i] - x2[j])**2)
                else:
                    raise ValueError("The kernel must be one of 'poly' or 'gaussian'")
        return K

    else:
        raise ValueError("x1 and x2 must both be 1D or 2D arrays")



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
            # Primal formulation for linear kernel
            self._w = np.zeros(n_features)
            self._b = 0.0

            for _ in range(self.n_iters):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                z_t = np.dot(self._w, x_t) + self._b
                # Gradient descent for logistic loss
                h_x = self._logistic(z_t)

                # Update weights and bias
                self._w -= self.learning_rate * (-(y_t - h_x) * x_t + self.lambda_param * self._w)
                self._b -= self.learning_rate * (-(y_t - h_x))

        elif self.kernel in ['poly', 'gaussian']:
            # Dual formulation with alpha coefficients
            # w = sum_i alpha_i * phi(x_i)
            # h(x) = 1 / (1 + exp(-(sum_i alpha_i * K(x_i, x) + b)))

            self._X_train = X
            self._y_train = y
            self._alpha = np.zeros(n_samples)
            self._b = 0.0

            # Convert labels to {0, 1} for logistic regression
            y_binary = (y + 1) / 2  # -1 -> 0, 1 -> 1

            for _ in range(self.n_iters):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t_binary = y_binary[idx]

                # Compute h(x) = 1 / (1 + exp(-(sum_i alpha_i * K(x_i, x) + b)))
                z = self._b
                for i in range(n_samples):
                    if abs(self._alpha[i]) > 1e-10:
                        z += self._alpha[i] * kernel_function(
                            X[i], x_t, kernel=self.kernel,
                            degree=self.degree, gamma=self.gamma
                        )

                h_x = self._logistic(z)

                # Gradient descent on alpha
                # Loss: -y*log(h) - (1-y)*log(1-h) + lambda/2 * ||alpha||^2
                # For each alpha_i: gradient = -(y - h) * K(x_i, x_t) + lambda * alpha_i
                error = h_x - y_t_binary

                for i in range(n_samples):
                    k_val = kernel_function(X[i], x_t, kernel=self.kernel,
                                           degree=self.degree, gamma=self.gamma)
                    gradient = error * k_val + self.lambda_param * self._alpha[i]
                    self._alpha[i] -= self.learning_rate * gradient

                # Update bias
                self._b -= self.learning_rate * error

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

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
            if self._alpha is None or self._X_train is None:
                raise ValueError("The model must be trained before any prediction")

            predictions = np.zeros(X.shape[0])

            for i in range(X.shape[0]):
                # h(x) = 1 / (1 + exp(-(sum_i alpha_i * K(x_i, x) + b)))
                z = self._b
                for j in range(len(self._X_train)):
                    if abs(self._alpha[j]) > 1e-10:
                        z += self._alpha[j] * kernel_function(
                            self._X_train[j], X[i], kernel=self.kernel,
                            degree=self.degree, gamma=self.gamma
                        )

                predictions[i] = self._logistic(z)

            return np.where(predictions >= 0.5, 1, -1)

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")