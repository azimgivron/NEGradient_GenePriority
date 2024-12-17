# pylint: disable=R0914
"""
Preprocessing module
=====================

Provides utilities for data preprocessing, including dataset index management, 
train-test splitting, data format conversions, and statistical analysis, 
streamlining data preparation workflows.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold, train_test_split

from NEGradient_GenePriority.preprocessing.train_test_indices import TrainTestIndices


def combine_splits(
    splits1: List[TrainTestIndices], splits2: List[TrainTestIndices]
) -> List[TrainTestIndices]:
    """
    Combines corresponding train-test splits from two lists of TrainTestIndices.

    Args:
        splits1 (List[TrainTestIndices]): The first list of TrainTestIndices.
        splits2 (List[TrainTestIndices]): The second list of TrainTestIndices.

    Returns:
        List[TrainTestIndices]: A new list of TrainTestIndices objects with
            combined training and testing indices.
    """
    return [split1.merge(split2) for split1, split2 in zip(splits1, splits2)]


def convert_dataframe_to_sparse_matrix(
    dataframe: pd.DataFrame, shape: Tuple[int, int]
) -> sp.coo_matrix:
    """
    Converts a DataFrame of pairwise associations into a COO sparse matrix.

    Args:
        dataframe (pd.DataFrame): DataFrame containing two columns of IDs representing
                                  associations.
        shape (Tuple[int, int]): The shape of the output matrix.

    Returns:
        sp.coo_matrix: Sparse matrix where rows and columns correspond to the
                       two ID columns in the DataFrame, and non-zero entries
                       indicate associations.
    """
    row, col = dataframe.to_numpy().T
    association_data = np.ones(len(row))
    return sp.coo_matrix((association_data, (row, col)), shape=shape)


def sample_zeros(
    sparse_matrix: sp.coo_matrix, sampling_factor: int, seed: int = None
) -> sp.coo_matrix:
    """
    Randomly samples zero entries from the complement of the sparse matrix's non-zero entries.

    Args:
        sparse_matrix (sp.coo_matrix): The input sparse matrix with existing non-zero entries.
        sampling_factor (int): The number of zero entries to sample, expressed as a multiple
                               of the current number of non-zero entries.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        sp.coo_matrix: Sparse matrix of sampled zeros.
    """
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Get the number of existing non-zero entries
    num_existing_ones = sparse_matrix.nnz
    num_sampled_zeros = num_existing_ones * sampling_factor

    # Get the matrix dimensions and existing non-zero indices
    total_rows, total_cols = sparse_matrix.shape
    non_zero_indices = set(zip(sparse_matrix.row, sparse_matrix.col))

    # Initialize sampling
    sampled_zeros = []
    sampled_count = 0

    while sampled_count < num_sampled_zeros:
        # Randomly sample row and column indices
        row_indices = np.random.randint(
            0, total_rows, size=num_sampled_zeros - sampled_count
        )
        col_indices = np.random.randint(
            0, total_cols, size=num_sampled_zeros - sampled_count
        )
        new_indices = set(zip(row_indices, col_indices))

        # Filter out indices that already exist in the sparse matrix
        unique_indices = [
            idx
            for idx in new_indices
            if idx not in non_zero_indices.union(set(sampled_zeros))
        ]

        # Add unique indices to the sampled set
        sampled_zeros.extend(unique_indices)
        sampled_count += len(unique_indices)

    # Convert sampled indices to sparse COO matrix
    sampled_zeros = np.array(sampled_zeros[:num_sampled_zeros])
    zero_data = np.zeros(len(sampled_zeros))
    result = sp.coo_matrix(
        (zero_data, (sampled_zeros[:, 0], sampled_zeros[:, 1])),
        shape=sparse_matrix.shape,
    )
    return result


def combine_matrices(matrix1: sp.coo_matrix, matrix2: sp.coo_matrix) -> sp.coo_matrix:
    """
    Combines two sparse matrices into a single sparse matrix.

    Args:
        matrix1 (sp.coo_matrix): The first sparse matrix.
        matrix2 (sp.coo_matrix): The second sparse matrix.

    Returns:
        sp.coo_matrix: The combined sparse matrix.
    """
    combined_row = np.hstack([matrix1.row, matrix2.row])
    combined_col = np.hstack([matrix1.col, matrix2.col])
    combined_data = np.hstack([matrix1.data, matrix2.data])

    return sp.coo_matrix(
        (combined_data, (combined_row, combined_col)), shape=matrix1.shape
    )


def filter_by_number_of_association(
    dataframe: pd.DataFrame, threshold: int, col_name: str
) -> pd.DataFrame:
    """
    Filters rows in a DataFrame based on a minimum number of associations.

    Args:
        dataframe (pd.DataFrame): DataFrame containing association data.
        threshold (int): Minimum number of associations required for a row to be retained.
        col_name (str): Name of the column to evaluate for filtering.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows that meet the threshold criteria.
    """
    association_counts = dataframe[col_name].value_counts()
    valid_ids = association_counts[association_counts >= threshold].index
    return dataframe[dataframe[col_name].isin(valid_ids)].reset_index(drop=True)


def create_random_splits_from_matrix(
    sparse_matrix: sp.coo_matrix, num_splits: int
) -> List[TrainTestIndices]:
    """
    Creates random train-test splits from an indices matrix.

    Args:
        sparse_matrix (sp.coo_matrix): Sparse matrix to be split into random subsets.
        num_splits (int): Number of random splits to create.

    Returns:
        List[TrainTestIndices]: List of TrainTestIndices objects containing
            training and testing subsets for each split.
    """
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).T
    splits = []
    for random_state in range(num_splits):
        train_idx, test_idx = train_test_split(
            indices,
            train_size=0.9,
            random_state=random_state,
            shuffle=True,
        )
        train_test_indices = TrainTestIndices.from_ndarray(train_idx, test_idx)
        splits.append(train_test_indices)
    return splits


def create_random_splits_from_matrices(
    sparse_matrix_list: List[sp.coo_matrix],
) -> List[TrainTestIndices]:
    """
    Creates random train-test splits from a list of indices matrices.

    Args:
        sparse_matrix_list (List[sp.coo_matrix]): List of sparse matrices to be split
            into random subsets.

    Returns:
        List[TrainTestIndices]: List of TrainTestIndices objects containing
            training and testing subsets for each split.
    """
    splits = []
    for sparse_matrix in sparse_matrix_list:
        splits.extend(create_random_splits_from_matrix(sparse_matrix, num_splits=1))
    return splits


def create_folds(
    sparse_matrix: sp.coo_matrix, num_folds: int
) -> List[TrainTestIndices]:
    """
    Splits a sparse matrix into K train-test folds using K-Fold cross-validation.

    Args:
        sparse_matrix (sp.coo_matrix): Sparse matrix to be split into folds.
        num_folds (int): Number of folds for the split.

    Returns:
        List[TrainTestIndices]: List of TrainTestIndices objects containing training
            and testing subsets for each fold.
    """
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).T
    return [
        TrainTestIndices.from_ndarray(indices[train_idx], indices[test_idx])
        for train_idx, test_idx in kfold.split(indices)
    ]


def compute_statistics(
    sparse_matrix: sp.coo_matrix, splits: List[TrainTestIndices]
) -> pd.DataFrame:
    """
    Computes summary statistics (mean, variance) of unique testing entries across splits.

    Args:
        sparse_matrix (sp.coo_matrix): The input sparse matrix.
        splits (List[TrainTestIndices]): List of TrainTestIndices objects.

    Returns:
        pd.DataFrame: DataFrame summarizing the counts and statistics of testing entries
                      across all splits.
    """
    counts = []
    for split in splits:
        data = split.testing_indices.get_data(sparse_matrix).tocoo()
        columns_with_ones = np.unique(data.col[data.data == 1])
        counts.append(len(columns_with_ones))
    average_count = np.mean(counts)
    variance_count = np.std(counts)
    return pd.DataFrame(
        [[*counts, average_count, variance_count]],
        columns=(
            [f"Split {i+1}" for i in range(len(counts))]
            + ["Average", "Standard Deviation"]
        ),
        index=["Counts"],
    ).T
