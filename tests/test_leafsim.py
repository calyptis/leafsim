"""Tests for leafsim.leafsim."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from leafsim import LeafSim, SUPPORTED_MODELS


@pytest.fixture
def iris():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train


@pytest.fixture
def fitted_classifier(iris):
    X_train, _, y_train = iris
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)


@pytest.fixture
def fitted_regressor(iris):
    X_train, _, y_train = iris
    return RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)


# --- Construction ---


def test_construction_classifier(fitted_classifier):
    ls = LeafSim(fitted_classifier)
    assert ls.model is fitted_classifier
    assert ls.model_name == "RandomForestClassifier"
    assert ls.index_func_params == {}


def test_construction_regressor(fitted_regressor):
    ls = LeafSim(fitted_regressor)
    assert ls.model_name == "RandomForestRegressor"


def test_construction_custom_index_func_params(fitted_classifier):
    params = {"check_input": False}
    ls = LeafSim(fitted_classifier, index_func_params=params)
    assert ls.index_func_params == params


def test_construction_unsupported_model_raises():
    class UnsupportedModel:
        pass

    with pytest.raises(TypeError, match="currently supported models"):
        LeafSim(UnsupportedModel())


def test_construction_custom_model_with_get_leaf_indices(iris, fitted_classifier):
    X_train, X_test, _ = iris

    class CustomModel:
        def get_leaf_indices(self, X, **kwargs):
            return fitted_classifier.apply(X)

    ls = LeafSim(CustomModel())
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=5)
    assert ids.shape == (X_test.shape[0], 5)


def test_supported_models_exported():
    assert "RandomForestClassifier" in SUPPORTED_MODELS
    assert "RandomForestRegressor" in SUPPORTED_MODELS


# --- generate_explanations ---


def test_output_shapes(fitted_classifier, iris):
    X_train, X_test, _ = iris
    top_n = 5
    ls = LeafSim(fitted_classifier)
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=top_n)
    assert ids.shape == (X_test.shape[0], top_n)
    assert sims.shape == (X_test.shape[0], top_n)


def test_similarity_range(fitted_classifier, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_classifier)
    _, sims = ls.generate_explanations(X_train, X_test, top_n=10)
    assert np.all(sims >= 0)
    assert np.all(sims <= 1)


def test_return_all_similarities(fitted_classifier, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_classifier)
    result = ls.generate_explanations(X_train, X_test, top_n=5, return_all_similarities=True)
    assert len(result) == 3
    ids, top_sims, all_sims = result
    assert all_sims.shape == (X_test.shape[0], X_train.shape[0])


def test_top_n_exceeds_train_size_raises(fitted_classifier, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_classifier)
    with pytest.raises(ValueError, match="top_n"):
        ls.generate_explanations(X_train, X_test, top_n=X_train.shape[0] + 1)


def test_regressor_output_shapes(fitted_regressor, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_regressor)
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=3)
    assert ids.shape == (X_test.shape[0], 3)
    assert sims.shape == (X_test.shape[0], 3)
