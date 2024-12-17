# pylint: disable=R0903
"""
TrainTestIndices module
=======================

This module defines the `TrainTestIndices` class, which encapsulates training and testing indices 
for dataset splitting and provides methods for interacting with these subsets.
"""

from __future__ import annotations

from typing import Type

import numpy as np

from NEGradient_GenePriority.preprocessing.indices import Indices


class TrainTestIndices:
    """
    Represents training and testing indices for dataset splitting.

    This class encapsulates two sets of indices: one for training and one for testing.
    It provides methods to interact with both subsets and retrieve the associated data
    from a dataset. It also supports merging with another TrainTestIndices object.

    Attributes:
        training_indices (Indices): An Indices object representing the training data indices.
        testing_indices (Indices): An Indices object representing the testing data indices.
    """

    def __init__(self, training_indices: Indices, testing_indices: Indices):
        """
        Initializes the TrainTestIndices object with training and testing indices.

        Args:
            training_indices (Indices): An Indices object for the training data.
            testing_indices (Indices): An Indices object for the testing data.
        """
        self.training_indices: Indices = training_indices
        self.testing_indices: Indices = testing_indices

    @classmethod
    def from_ndarray(
        cls: Type[TrainTestIndices],
        training_indices: np.ndarray,
        testing_indices: np.ndarray,
    ) -> TrainTestIndices:
        """
        Creates a TrainTestIndices instance from two numpy arrays.

        Args:
            training_indices (np.ndarray): A 2D array of shape (n, 2) containing
                training data indices.
            testing_indices (np.ndarray): A 2D array of shape (m, 2) containing
                testing data indices.

        Returns:
            TrainTestIndices: A TrainTestIndices instance initialized with the given indices.
        """
        return cls(Indices(training_indices), Indices(testing_indices))

    @classmethod
    def from_indices(
        cls: Type[TrainTestIndices], training_indices: Indices, testing_indices: Indices
    ) -> TrainTestIndices:
        """
        Creates a TrainTestIndices instance directly from Indices objects.

        Args:
            training_indices (Indices): An Indices object for the training data.
            testing_indices (Indices): An Indices object for the testing data.

        Returns:
            TrainTestIndices: A TrainTestIndices instance initialized with the
                given Indices objects.
        """
        return cls(training_indices, testing_indices)

    def merge(self, train_test_indices: TrainTestIndices) -> TrainTestIndices:
        """
        Merges another TrainTestIndices object into the current one.

        Args:
            train_test_indices (TrainTestIndices): Another TrainTestIndices
                object to merge.

        Returns:
            TrainTestIndices: A new TrainTestIndices object with merged training
                and testing indices.
        """
        merged_training = self.training_indices.merge(
            train_test_indices.training_indices
        )
        merged_testing = self.testing_indices.merge(train_test_indices.testing_indices)
        return TrainTestIndices.from_indices(merged_training, merged_testing)
