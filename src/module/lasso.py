import numpy as np


class Lasso:
    """
    Implementation of Lasso regression using proximal gradient method.

    Attributes
    ----------
    alpha : float
        The regularization parameter.
    max_iter : int
        Maximum number of iterations for the proximal gradient method.
    tol : float
        The tolerance for the optimization.
    coef_ : ndarray
        The estimated coefficients for the linear regression model.
    intercept_ : float
        The estimated intercept (bias) term.
    """
    def __init__(self, alpha=0.1, max_iter=1000, tol=0.0001):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, x, lambda_):
        """_summary_

        Args:
            x (_type_): _description_
            lambda_ (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def _update_coef(self, X, y, L):
        """"""
        n = X.shape[0]
        grad = (X.T @ (X @ self.coef_ + self.intercept_ - y) + n * self.intercept_) / n
        self.coef_ = self._soft_threshold(self.coef_ - grad / L, self.alpha / L)

    def fit(self, X, y):
        n, d = X.shape
        self.coef_ = np.zeros(d)
        self.intercept_ = 0
        L = np.linalg.norm(X, 2) ** 2 / n  # Lipshitz constant

        for _ in range(self.max_iter):
            coef_prev = self.coef_.copy()
            intercept_prev = self.intercept_

            # Proximal gradient descent for Lasso
            self._update_coef(X, y, L)

            residual = y - np.dot(X, self.coef_)
            self.intercept_ = np.mean(residual)

            # Check for convergence
            if np.max(np.abs(coef_prev - self.coef_)) < self.tol and np.abs(intercept_prev - self.intercept_) < self.tol:
                break

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y_mean) ** 2).sum()
        return 1 - u / v
