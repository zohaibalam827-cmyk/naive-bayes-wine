import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.data import load_data
from src.naive_bayes import GaussianNaiveBayes


def test_separates_two_clusters():
    X = [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]
    y = ["a", "a", "a", "b", "b", "b"]
    model = GaussianNaiveBayes().fit(X, y)
    assert list(model.predict([[1.5], [10.5]])) == ["a", "b"]


def test_priors_match_class_frequencies():
    X = [[0.0], [1.0], [2.0], [3.0], [10.0], [11.0]]
    y = ["a", "a", "a", "a", "b", "b"]
    model = GaussianNaiveBayes().fit(X, y)
    assert list(model.classes_) == ["a", "b"]
    assert np.allclose(model.priors_, [4 / 6, 2 / 6])


def test_probabilities_sum_to_one():
    X, y, _ = load_data()
    model = GaussianNaiveBayes().fit(X, y)
    probs = model.predict_proba(X[:20])
    assert probs.shape == (20, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_constant_feature_does_not_break():
    X = [[1.0, 5.0], [2.0, 5.0], [10.0, 5.0], [11.0, 5.0]]
    y = ["a", "a", "b", "b"]
    model = GaussianNaiveBayes().fit(X, y)
    assert np.all(np.isfinite(model.predict_proba(X)))
    assert list(model.predict(X)) == y


def test_invalid_inputs():
    with pytest.raises(ValueError):
        GaussianNaiveBayes(var_smoothing=-1)
    with pytest.raises(ValueError):
        GaussianNaiveBayes().fit([[1.0], [2.0]], ["a"])
    with pytest.raises(ValueError):
        GaussianNaiveBayes().fit([], [])
    with pytest.raises(RuntimeError):
        GaussianNaiveBayes().predict([[1.0]])
    model = GaussianNaiveBayes().fit([[1.0, 2.0], [3.0, 4.0]], ["a", "b"])
    with pytest.raises(ValueError):
        model.predict([[1.0]])


def test_matches_scikit_learn_on_wine():
    X, y, _ = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    ours = GaussianNaiveBayes().fit(X_tr, y_tr)
    reference = GaussianNB().fit(X_tr, y_tr)
    assert ours.score(X_te, y_te) > 0.9
    assert abs(ours.score(X_te, y_te) - reference.score(X_te, y_te)) < 0.02
