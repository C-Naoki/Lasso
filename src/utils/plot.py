import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

from ..module.lasso import Lasso


def plot_coef_path(X, y, alphas):
    n_features = X.shape[1]
    coef_path = np.zeros((len(alphas), n_features))

    # Compute coefficients for each alpha
    for i, alpha in enumerate(alphas):
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        coef_path[i] = lasso.coef_

    # Plot coefficient path
    plt.figure(figsize=(10, 6))
    for feature in range(n_features):
        plt.plot(alphas, coef_path[:, feature], label=f'Feature {feature}')

    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Standardized Coefficients')
    plt.title('Coefficient Paths of Features')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Make regression data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define alphas
    alphas = np.logspace(-2, 2.5, num=100)

    # Plot coefficient path
    plot_coef_path(X, y, alphas)
