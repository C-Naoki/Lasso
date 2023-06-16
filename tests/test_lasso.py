from time import time

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as SklearnLasso

from src.module.lasso import Lasso


def test_lasso() -> None:
    """
    Test the custom Lasso regression implementation against sklearn's implementation.
    This function generates a random regression problem, fits the model using both
    custom and sklearn's Lasso classes, checks that the scores are close, and
    compares the computation time.
    """
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1)
    l1 = Lasso(alpha=0.1)
    l2 = SklearnLasso(alpha=0.1)

    start = time()
    l1.fit(X, y)
    time_custom = time() - start

    start = time()
    l2.fit(X, y)
    time_sklearn = time() - start

    score1 = l1.score(X, y)
    score2 = l2.score(X, y)
    assert abs(score1 - score2) < 1e-2, f"Expected {score2}, but got {score1}"

    plt.bar(["Custom", "Sklearn"], [time_custom, time_sklearn])
    plt.ylabel("Time (s)")
    plt.title("Computation Time Comparison between Custom and Sklearn Lasso")
    plt.savefig("logs/fig/computation_time.pdf")
    plt.show()
