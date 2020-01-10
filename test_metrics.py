"""Test cases for ranking metrics."""
import numpy as np

from src.metrics import average_precision_at_k, dcg_at_k, recall_at_k

y_true: np.ndarray = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 0])
y_score: np.ndarray = np.array([1, 2, 3, 4, 5, 6, 7, 8,  9, 10])[::-1]
propensity: np.ndarray = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
k: int = 4


def test_dcg_at_k_01() -> None:
    """Test case1 for dcg_at_k."""
    assert dcg_at_k(y_true=y_true, y_score=y_score, k=k) == \
        (1 + (np.log2(4)) ** (-1)) / 4


def test_dcg_at_k_02() -> None:
    """Test case2 for dcg_at_k."""
    assert dcg_at_k(
        y_true=y_true, y_score=y_score, k=k, propensity=propensity
    ) == (10 + 2.5 * (np.log2(4)) ** (-1)) / 15.75


def test_recall_at_k_01() -> None:
    """Test case1 for recall_at_k."""
    assert recall_at_k(y_true=y_true, y_score=y_score, k=k) == 0.5


def test_recall_at_k_02() -> None:
    """Test case2 for recall_at_k."""
    assert recall_at_k(
        y_true=y_true, y_score=y_score, k=k, propensity=propensity,
    ) == 12.5 / 15.75


def test_average_precision_k_01() -> None:
    """Test case1 for average_precision_at_k."""
    assert average_precision_at_k(
        y_true=y_true, y_score=y_score, k=k) == 1.5 / 4


def test_average_precision_at_k_01() -> None:
    """Test case2 for average_precision_at_k."""
    assert average_precision_at_k(
        y_true=y_true, y_score=y_score, k=k, propensity=propensity,
    ) == 13.125 / 15.75
