"""
ModelEvaluationCollection module
===============================

This module provides the `ModelEvaluationCollection` class, which encapsulates the
evaluation results of multiple models. It offers utility methods to compute and
aggregate evaluation metrics such as AUC loss and BEDROC scores across different models.
This class facilitates comparison and analysis of model performance in a structured
and efficient manner.

"""
from typing import Dict, Iterator, List, Tuple

import numpy as np

from NEGradient_GenePriority.evaluation.evaluation import Evaluation


class ModelEvaluationCollection:
    """
    A collection of evaluation results for multiple models, with methods to compute
    and access aggregated metrics such as AUC loss and BEDROC scores.

    Attributes:
        model_results (Dict[str, Evaluation]): A dictionary where keys are model names
            and values are Evaluation objects containing metrics for the respective model.
    """

    def __init__(self, model_results: Dict[str, Evaluation]):
        """
        Initializes the ModelEvaluationCollection.

        Args:
            model_results (Dict[str, Evaluation]): A dictionary mapping model names
                to their respective Evaluation objects.
        """
        for key, val in model_results.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Invalid dictionary key: Expected `str`, but got {type(key)} for key "
                    f"`{key}`. Ensure all keys in `model_results` are strings."
                )
            if not isinstance(val, Evaluation):
                raise TypeError(
                    f"Invalid dictionary value for key `{key}`: Expected `Evaluation`, "
                    f"but got {type(val)}. Ensure all values in `model_results` are instances "
                    "of the `Evaluation` class."
                )
        self.model_results = model_results

    @property
    def model_names(self) -> List[str]:
        """
        Retrieves the names of the models in the collection.

        Returns:
            List[str]: A list of model names.
        """
        return list(self.model_results.keys())

    @property
    def evaluations(self) -> List[Evaluation]:
        """
        Retrieves the list of Evaluation objects in the collection.

        Returns:
            List[Evaluation]: A list of Evaluation objects corresponding to the models.
        """
        return list(self.model_results.values())

    def items(self) -> Iterator[Tuple[str, Evaluation]]:
        """
        Provides an iterator over the model names and their corresponding Evaluation objects.

        Returns:
            Iterator[Tuple[str, Evaluation]]: An iterator of (model_name, Evaluation) pairs.
        """
        return self.model_results.items()

    def __iter__(self) -> Iterator[Evaluation]:
        """
        Allows the `ModelEvaluationCollection` object to be iterable over its Evaluation objects.

        Returns:
            Iterator[Evaluation]: An iterator over the Evaluation objects.
        """
        return iter(self.evaluations)

    def compute_auc_losses(self) -> np.ndarray:
        """
        Calculates the mean and standard deviation of the
        AUC loss (1 - AUC) for each model across all diseases.

        Returns:
            np.ndarray: A 2D array containing the average and standard deviation
                of the AUC loss for each model. Shape: (models, 2).
        """
        auc_loss = np.array(
            [eval_res.compute_avg_auc_loss() for eval_res in self.evaluations]
        )
        avg_auc_loss = np.hstack((np.mean(auc_loss, axis=1), np.std(auc_loss, axis=1)))
        return avg_auc_loss.reshape((len(self.evaluations), 2))

    def compute_bedroc_scores(self) -> np.ndarray:
        """
        Calculates the BEDROC scores for several alpah values per disease and
        for each model.

        Returns:
            np.ndarray: A 3D array containing the BEDROC scores for each disease,
            across different alpha values, for each model. Shape: (alphas, diseases, models).
        """
        bedroc = np.array(
            [eval_res.compute_bedroc_scores() for eval_res in self.evaluations]
        )  # shape = (models, diseases, alphas)
        return bedroc.T
