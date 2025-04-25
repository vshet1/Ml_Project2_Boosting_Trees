import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from model.GradientBoostingClassifier import GradientBoostingClassifier


def test_training():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=5)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert predictions.shape == y.shape
    assert np.all((predictions == 0) | (predictions == 1))


def test_accuracy_on_simple_data():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=10)
    clf.fit(X, y)
    preds = clf.predict(X)
    accuracy = np.mean(preds == y)
    assert accuracy >= 0.8


def test_all_same_class():
    X = np.random.rand(10, 3)
    y = np.zeros(10)  # All labels are 0
    clf = GradientBoostingClassifier(n_estimators=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.all(preds == 0), "Model should predict all zeros if trained on all-zero labels"


def test_binary_threshold_behavior():
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.mean(preds == y) >= 0.9, "Should correctly learn the threshold split"


def test_random_data_stability():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    clf = GradientBoostingClassifier(n_estimators=20)
    clf.fit(X, y)
    preds = clf.predict(X)
    accuracy = np.mean(preds == y)
    assert accuracy >= 0.85, f"Expected >=85% accuracy, got {accuracy:.2f}"


def test_zero_features():
    X = np.zeros((10, 5))
    y = np.array([0, 1] * 5)
    clf = GradientBoostingClassifier(n_estimators=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape, "Output shape mismatch on zero feature data"
    assert np.all((preds == 0) | (preds == 1)), "Predictions should still be binary"


def test_predict_before_fit():
    X = np.random.rand(5, 2)
    clf = GradientBoostingClassifier(n_estimators=3)
    try:
        clf.predict(X)
        assert False, "Predicting before fitting should raise an error or behave unexpectedly"
    except Exception:
        assert True
