import numpy as np


def kernel_function(x1, x2, kernel='poly', degree=2, gamma=1.0):
    if kernel == 'poly':
        return (1 + np.dot(x1, x2))**degree
    elif kernel == 'gaussian':
        return np.exp(-1 / (2 * gamma) * np.linalg.norm(x1 - x2)**2)
    else:
        raise ValueError("The kernel must be one of 'poly' or 'gaussian'")



class SVM:
    def __init__(self, n_iters=1000, lambda_param=0.01, random_state=42, kernel='linear', degree=2, gamma=1.0, track_loss=False, loss_interval=10):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.random_state = random_state
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.track_loss = track_loss
        self.loss_interval = loss_interval
        self._w = None
        self._b = None
        self._w_avg = None
        self._b_avg = None
        self._alpha = []
        self._support_vectors = []
        self._support_labels = []
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)
        self.loss_history = []

        if self.kernel == 'linear':
            self._w = np.zeros(n_features)
            self._b = 0.0

            self._w_avg = np.zeros(n_features)
            self._b_avg = 0.0

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


                self._w_avg += self._w
                self._b_avg += self._b

                # Track loss at specified intervals
                if self.track_loss and t % self.loss_interval == 0:
                    loss = self.compute_loss(X, y)
                    self.loss_history.append((t, loss))

            # Compute final average
            self._w_avg /= self.n_iters
            self._b_avg /= self.n_iters

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

                # Track loss at specified intervals
                if self.track_loss and t % self.loss_interval == 0:
                    loss = self._compute_loss_kernel(X, y)
                    self.loss_history.append((t, loss))

    def compute_loss(self, X, y):
        """
        Compute the SVM loss (only for linear kernel).
        Loss = λ/2 ||w||² + 1/n Σ max(0, 1 - y_i(w·x_i + b))
        """
        if self.kernel != 'linear':
            raise NotImplementedError("Loss computation is only implemented for linear kernel")

        if self._w is None:
            raise ValueError("The model must be trained before computing the loss")

        n_samples = X.shape[0]

        # Regularization term: λ/2 ||w||²
        reg_term = (self.lambda_param / 2) * np.dot(self._w, self._w)

        # Hinge loss: 1/n Σ max(0, 1 - y_i(w·x_i + b))
        margins = y * (np.dot(X, self._w) + self._b)
        hinge_losses = np.maximum(0, 1 - margins)
        empirical_loss = np.mean(hinge_losses)

        return reg_term + empirical_loss

    def _compute_loss_kernel(self, X, y):
        """
        Compute the SVM loss for kernel methods (approximate).
        For non-linear kernels, computes empirical hinge loss only.
        """
        if self.kernel == 'linear':
            return self.compute_loss(X, y)

        # For kernel methods, compute empirical hinge loss
        predictions_scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            g = sum(
                alpha * kernel_function(sv, X[i], self.kernel, self.degree, self.gamma)
                for alpha, sv in zip(self._alpha, self._support_vectors)
            )
            predictions_scores[i] = g

        margins = y * predictions_scores
        hinge_losses = np.maximum(0, 1 - margins)
        return np.mean(hinge_losses)

    def predict(self, X):
        if self.kernel == 'linear':
            if self._w is None:
                raise ValueError("The model must be trained before any prediction")

            w = self._w_avg
            b = self._b_avg

            return np.sign(np.dot(X, w) + b)

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

    def __init__(self, n_iters=1000, lambda_param=0.01, kernel='linear', degree=2, gamma=1.0, random_state=42, track_loss=False, loss_interval=10):
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state
        self.track_loss = track_loss
        self.loss_interval = loss_interval
        self._w = None
        self._b = None
        self._w_avg = None
        self._b_avg = None
        self._alpha = None
        self._X_train = None
        self._y_train = None
        self.loss_history = []

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if not np.array_equal(np.sort(np.unique(y)), np.array([-1, 1])):
            raise ValueError("y must contain only -1 and 1 values")

        n_samples, n_features = X.shape
        np.random.seed(self.random_state)
        self.loss_history = []

        if self.kernel == 'linear':
            self._w = np.zeros(n_features)
            self._b = 0.0

            self._w_avg = np.zeros(n_features)
            self._b_avg = 0.0

            for t in range(1, self.n_iters + 1):
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                z_t = y_t * (np.dot(self._w, x_t) + self._b)

                eta = 1 / (self.lambda_param * t)
                sigma_term = self._logistic(-z_t)

                # Update w with regularization, b without
                self._w = (1 - eta * self.lambda_param) * self._w + \
                        eta * sigma_term * y_t * x_t

                self._b = self._b + eta * sigma_term * y_t

                # Update running average
                self._w_avg += self._w
                self._b_avg += self._b

                # Track loss at specified intervals
                if self.track_loss and t % self.loss_interval == 0:
                    loss = self.compute_loss(X, y)
                    self.loss_history.append((t, loss))

            # Compute final average
            self._w_avg /= self.n_iters
            self._b_avg /= self.n_iters

        elif self.kernel in ['poly', 'gaussian']:
            self._alpha = []
            self._support_vectors = []
            
            for t in range(1, self.n_iters + 1):
                # Random sample selection
                idx = np.random.randint(0, n_samples)
                x_t = X[idx]
                y_t = y[idx]

                # Learning rate
                eta = 1 / (self.lambda_param * t)

                # Compute g_t(x_t)
                g_t = sum(
                    alpha * kernel_function(sv, x_t, self.kernel, self.degree, self.gamma)
                    for alpha, sv in zip(self._alpha, self._support_vectors)
                )

                # Continuous weight: σ(-y_t * g_t)
                weight = self._logistic(-y_t * g_t)

                self._alpha = [(1 - eta * self.lambda_param) * a for a in self._alpha]

                # Add only if weight is significant (analogous to margin condition in SVM)
                # weight = σ(-y_t * g_t) is high when the model is uncertain or makes an error
                if weight > self._WEIGHT_THRESHOLD:
                    self._support_vectors.append(x_t.copy())
                    self._alpha.append((y_t / (self.lambda_param * t)) * weight)


    def _logistic(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y):
        """
        Compute the Logistic Regression loss (only for linear kernel).
        Loss = λ/2 ||w||² + 1/n Σ log(1 + exp(-y_i * (w·x_i + b)))
        """
        if self.kernel != 'linear':
            raise NotImplementedError("Loss computation is only implemented for linear kernel")

        if self._w is None:
            raise ValueError("The model must be trained before computing the loss")

        # Regularization term: λ/2 ||w||²
        reg_term = (self.lambda_param / 2) * np.dot(self._w, self._w)

        # Logistic loss: 1/n Σ log(1 + exp(-y_i * (w·x_i + b)))
        z = y * (np.dot(X, self._w) + self._b)
        # Use log1p for numerical stability: log(1 + exp(-z)) = log1p(exp(-z))
        # For large negative z, exp(-z) is very large, so we use the identity:
        # log(1 + exp(-z)) = -z + log(1 + exp(z)) for z < 0
        logistic_losses = np.where(
            z >= 0,
            np.log1p(np.exp(-z)),
            -z + np.log1p(np.exp(z))
        )
        empirical_loss = np.mean(logistic_losses)

        return reg_term + empirical_loss

    def predict(self, X):
        if self.kernel == 'linear':
            if self._w is None:
                raise ValueError("The model must be trained before any prediction")

            # Use averaged weights if averaging is enabled

            w = self._w_avg
            b = self._b_avg

            predictions = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                z = np.dot(w, X[i]) + b
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