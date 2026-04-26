"""
LeafSim — example-based explanations for tree-based ensemble models.

For a given prediction, LeafSim identifies the training samples most similar to
the sample being explained. Similarity is measured by counting how many trees in
the ensemble assign both samples to the same leaf node (Hamming distance over leaf
indices). A high score means the two samples follow the same decision paths through
the forest, making them naturally comparable for explanation purposes.
"""
import logging
from typing import Optional, Union

import numpy as np
from sklearn.metrics import DistanceMetric

logger = logging.getLogger("leafsim")
logger.addHandler(logging.NullHandler())

# Functions that get leaf indices for the different models supported by LeafSim
# E.g. catboost models use calc_leaf_indexes() while sklearn models use apply()
LEAF_INDEX_FUNC = {
    "CatBoostRegressor": "calc_leaf_indexes",
    "CatBoostClassifier": "calc_leaf_indexes",
    "RandomForestRegressor": "apply",
    "RandomForestClassifier": "apply",
    "XGBRegressor": "apply",
    "XGBClassifier": "apply",
}
# Default parameters to use for the models supported by LeafSim
LEAF_INDEX_DEFAULT_PARAMS = {
    "CatBoostRegressor": {
        "ntree_start": 0,
        "ntree_end": 0,
        "thread_count": -1,
        "verbose": False,
    },
    "CatBoostClassifier": {
        "ntree_start": 0,
        "ntree_end": 0,
        "thread_count": -1,
        "verbose": False,
    },
    "RandomForestRegressor": {},
    "RandomForestClassifier": {},
    "XGBRegressor": {},
    "XGBClassifier": {},
}

SUPPORTED_MODELS = sorted(list(LEAF_INDEX_FUNC.keys()))


class LeafSim:
    """LeafSim class."""

    def __init__(self, model, index_func_params: Optional[dict] = None):
        """
        Initialise the LeafSim instance.

        This defines the ML model that we want to explain.
        It also specifies the function to identify what
        observations fall into which leaves.

        :param model: Tree-based ensemble model, one from LEAF_INDEX_FUNC.keys()
        :param index_func_params: Parameters passed onto the leaf indexing function
        """
        # Set model
        self.model = model
        self.model_name = str(self.model.__class__.__name__)

        # Get leaf indexing function
        index_func = LEAF_INDEX_FUNC.get(self.model_name, None)
        if index_func is None:
            # If providing a model that is not supported by LeafSim out of the box
            # This new model needs to have an attribute "get_leaf_indices"
            try:
                index_func = self.model.get_leaf_indices
            except AttributeError:
                supported = "\n".join(SUPPORTED_MODELS)
                error_msg = (
                    f"Provide one of the following currently supported models:\n\n"
                    f"{supported}\n\n"
                    f"or provide a custom model instance with a get_leaf_indices attribute.\n"
                    f"This must be a function that returns leaf indices as a matrix of shape [n_samples, n_predictors]."
                )
                raise TypeError(error_msg)
        else:
            index_func = self.model.__getattribute__(index_func)
        self.index_func = index_func

        # Get leaf indexing function parameters
        if index_func_params is None:
            self.index_func_params = LEAF_INDEX_DEFAULT_PARAMS.get(self.model_name, {})
        else:
            self.index_func_params = index_func_params

    def get_leaf_indices(self, X: np.ndarray, params: Optional[dict] = None):
        """
        Get the indices of leaves for every observation in the feature matrix X.

        The function gets one index for every observation and tree in the ensemble model.

        :param X: feature matrix
        :param params:
            These parameters passed onto the function that gets the indices.
            Supported values depend on the model one wishes to generate explanations for.
        :return leaf_indices: Indices of the leaves in the shape of (X.shape[0], # leaves)
        """
        if params is not None:
            self.index_func_params = params

        # Get a matrix with each row containing the leaf indices
        # across all trees for a given instance
        leaf_indices = self.index_func(X, **self.index_func_params)

        return leaf_indices

    def generate_explanations(
        self,
        X_train: np.ndarray,
        X_to_explain: np.ndarray,
        params: Optional[dict] = None,
        top_n: int = 10,
        return_all_similarities: bool = False,
    ) -> Union[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Identify the training examples that explain the prediction of X_to_explain.

        :param X_train: Data the model to explain was trained on.
        :param X_to_explain: Data points one wished to generate explanations for.
        :param params: Parameters for the function that returns
                       indices of leaves for every observation in X_to_explain.
                       See official documentation of the LEAF_INDEX_FUNC functions
                       supported by LeafSim.
        :param top_n: The number of explanations to provide.
                      By default, provide the 10 closest matches to every
                      observation in X_to_explain.
        :param return_all_similarities: Whether to return the similarities for
                                        all training observations.
        :return top_n_ids: Integer location for the observations in X_train that are among
                           the top_n.
        :return top_n_similarity: The corresponding Hamming distance of the observation
                                  in the top_n_ids and the observation one wishes
                                  to generate an explanation for.
        """
        if top_n > X_train.shape[0]:
            raise ValueError(
                f"top_n ({top_n}) cannot exceed the number of training samples ({X_train.shape[0]})"
            )
        logger.info("Getting leaf indices of samples in training data")
        train_leaf_indices = self.get_leaf_indices(X_train, params)
        logger.info("Getting leaf indices of samples in test data")
        test_leaf_indices = self.get_leaf_indices(X_to_explain, params)
        logger.info("Measuring distances between every train and test data point")
        distances = DistanceMetric.get_metric("hamming").pairwise(
            X=test_leaf_indices, Y=train_leaf_indices
        )
        logger.info(f"Identifying top {top_n} most similar training data points for each test data point")
        sorted_distances = np.argsort(distances, axis=1)
        # For each instance we want to explain, select only
        # the top N similar training instances
        # Shape: # test samples, Top N most similar train samples
        top_n_ids = sorted_distances[:, :top_n]

        # For the top N most similar training instances,
        # obtain their corresponding similarity score
        # Shape: # test samples, similarity of Top N train samples
        row_idx = np.arange(distances.shape[0])[:, None]
        top_n_similarity = 1 - distances[row_idx, top_n_ids]

        if return_all_similarities:
            similarities = 1 - distances
            return top_n_ids, top_n_similarity, similarities
        else:
            return top_n_ids, top_n_similarity
