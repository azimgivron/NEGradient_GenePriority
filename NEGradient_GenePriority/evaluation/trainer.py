# pylint: disable=R0913,R0914,R0902
"""
Trainer module
=================

This module orchestrates the training and evaluation of predictive models for gene
prioritization tasks. It integrates preprocessing and evaluation functionalities
to compute and log performance metrics across train-test splits or cross-validation
folds.

This module supports efficient and reproducible workflows by utilizing a `DataLoader`
for preprocessing, and saving evaluation results for further analysis.
"""

import logging
import pickle
from typing import Dict, List, Literal, Tuple

import numpy as np
import smurff
from tqdm import tqdm

from NEGradient_GenePriority.evaluation.evaluation import Evaluation
from NEGradient_GenePriority.evaluation.results import Results
from NEGradient_GenePriority.preprocessing.dataloader import DataLoader
from NEGradient_GenePriority.preprocessing.side_information_loader import (
    SideInformationLoader,
)


class Trainer:
    """
    Trainer Class

    This class manages the training and evaluation of predictive models for gene
    prioritization tasks. It supports evaluation across train-test splits or
    cross-validation folds and integrates preprocessing and metrics functionalities
    to compute and log performance.

    Attributes:
        dataloader (DataLoader): The data loader containing all data for training
            and testing.
        side_info_loader (SideInformationLoader): The data loader for the
            side information.
        path (str): The path to the directory where model snapshots will be saved.
        num_samples (int): The number of posterior samples to draw during training.
        burnin_period (int): The number of burn-in iterations before collecting
            posterior samples.
        direct (bool): Whether to use a Cholesky or conjugate gradient (CG) solver.
        univariate (bool): Whether to use univariate or multivariate sampling.
        seed (int): The random seed for reproducibility.
        save_freq (int): Frequency at which model state is saved (e.g., every N samples).
        verbose (Literal[0, 1, 2]): Verbosity level of the algorithm
            (0: Silent, 1: Minimal, 2: Detailed).
        logger (logging.Logger): Logger instance for debug and info messages.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        path: str,
        num_samples: int,
        burnin_period: int,
        direct: bool,
        univariate: bool,
        seed: int,
        save_freq: int,
        verbose: Literal[0, 1, 2],
        side_info_loader: SideInformationLoader = None,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize the Trainer class with the given configuration.

        Args:
            dataloader (DataLoader): The data loader containing all data for
                training and testing.
            path (str): The path to the directory where model snapshots will be saved.
            num_samples (int): The number of posterior samples to draw during training.
            burnin_period (int): The number of burn-in iterations before collecting
                posterior samples.
            direct (bool): Whether to use a Cholesky or CG solver.
            univariate (bool): Whether to use univariate or multivariate sampling.
            seed (int): The random seed for reproducibility.
            save_freq (int): The frequency at which the model state is saved.
            verbose (Literal[0, 1, 2]): The verbosity level of the algorithm.
            side_info_loader (SideInformationLoader, optional): The data loader for the
                side information.
            logger (logging.Logger, optional): Logger instance for debug messages.
                If None, a default logger is created.
        """
        self.dataloader = dataloader
        self.side_info_loader = side_info_loader
        self.path = path
        self.num_samples = num_samples
        self.burnin_period = burnin_period
        self.direct = direct
        self.univariate = univariate
        self.seed = seed
        self.save_freq = save_freq
        self.verbose = verbose

        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    @property
    def macau_session_kwargs(self) -> Dict[str, any]:
        """
        Generate keyword arguments for configuring the SMURFF Macau session.

        Returns:
            Dict[str, any]: A dictionary of parameters for the Macau session,
                including solver type, burn-in iterations, posterior sampling
                parameters, and verbosity settings.
        """
        return {
            "is_scarce": True,
            "direct": self.direct,
            "univariate": self.univariate,
            "burnin": self.burnin_period,
            "nsamples": self.num_samples,
            "seed": self.seed,
            "save_freq": self.save_freq,
            "verbose": self.verbose,
        }

    def __call__(
        self,
        latent_dimensions: List[int],
        save_results: bool,
        omim2_filename: str = "omim2_results.pickle",
        omim1_filename: str = "omim1_results.pickle",
    ) -> Tuple[Dict[str, Evaluation], Dict[str, Evaluation]]:
        """
        Execute training for different latent dimensions and optionally save results.

        Args:
            latent_dimensions (List[int]): List of latent dimensions to evaluate.
            save_results (bool): Whether to save the results to file.
            omim2_filename (str, optional): Filename for saving OMIM2 results.
                Defaults to "omim2_results.pickle".
            omim1_filename (str, optional): Filename for saving OMIM1 results.
                Defaults to "omim1_results.pickle".

        Returns:
            Tuple[Dict[str, Evaluation], Dict[str, Evaluation]]: Evaluation results
                for OMIM1 and OMIM2 datasets.
        """
        omim1_results = {}
        omim2_results = {}
        for num_latent in tqdm(latent_dimensions, desc="Latent dimensions"):
            self.logger.debug("Running MACAU for %d latent dimensions", num_latent)
            omim1_results[f"latent dim={num_latent}"] = self.train_test_splits(
                num_latent=num_latent,
                save_name=f"latent={num_latent}:macau-omim1.hdf5",
            )
            omim2_results[
                f"latent dim={num_latent}"
            ] = self.train_test_cross_validation(
                num_latent=num_latent,
                save_name=f"latent={num_latent}:macau-omim2.hdf5",
            )
        self.logger.debug("MACAU session completed successfully")
        if save_results:
            omim2_results_path = self.path / omim2_filename
            omim1_results_path = self.path / omim1_filename
            save_evaluations(omim2_results, omim2_results_path)
            save_evaluations(omim1_results, omim1_results_path)
            self.logger.debug("Results serialization completed successfully")
        return omim1_results, omim2_results

    def predict(
        self,
        session: smurff.MacauSession,
    ) -> np.ndarray:
        """
        Extract predictions from the trained model.

        Args:
            session (smurff.MacauSession): The trained SMURFF session.

        Returns:
            (np.ndarray): The predictions.
        """
        predict_session = session.makePredictSession()
        y_pred = np.mean(predict_session.predict_all(), axis=0)
        return y_pred

    def add_side_info(self, session: smurff.MacauSession):
        """
        Add side information to the SMURFF Macau session.

        Args:
            session (smurff.MacauSession): The SMURFF Macau session to which
                the side information will be added.
        """
        for disease_side_info in self.side_info_loader.disease_side_info:
            # The direct method is only feasible for a small (< 100K) number of
            # features.
            session.addSideInfo(mode=1, Y=disease_side_info, direct=False)
        for gene_side_info in self.side_info_loader.gene_side_info:
            session.addSideInfo(mode=0, Y=gene_side_info, direct=False)

    def train_test_cross_validation(
        self,
        num_latent: int,
        save_name: str,
    ) -> Evaluation:
        """
        Train and evaluate the model using cross-validation.

        Args:
            num_latent (int): The number of latent dimensions for the model.
            save_name (str): The base filename for saving model snapshots.

        Returns:
            Evaluation: Aggregated evaluation results across all folds.
        """
        results = []
        for i, (y_train, y_test_1s) in tqdm(
            enumerate(zip(*self.dataloader.folds)), desc="Folds", leave=False
        ):
            self.logger.debug("Initiating training on fold %d", i + 1)
            self.logger.debug(
                "Number of 1s in the training set %s",
                len(y_train.data[y_train.data == 1]),
            )
            self.logger.debug(
                "Number of 0s in the training set %s",
                len(y_train.data[y_train.data == 0]),
            )
            session = smurff.MacauSession(
                **self.macau_session_kwargs,
                num_latent=num_latent,
                Ytrain=y_train,
                Ytest=y_test_1s,
                save_name=str(self.path / f"{i}:{save_name}"),
                side_info=self.side_info_loader.side_info
                if self.side_info_loader
                else None,
            )
            session.run()
            y_pred = self.predict(session)
            results.append(Results(y_true=self.dataloader.omim2.tocsr(), y_pred=y_pred))
        return Evaluation(results)

    def train_test_splits(
        self,
        num_latent: int,
        save_name: str,
    ) -> Evaluation:
        """
        Train and evaluate the model using predefined train-test splits.

        Args:
            num_latent (int): The number of latent dimensions for the model.
            save_name (str): The base filename for saving model snapshots.

        Returns:
            Evaluation: Aggregated evaluation results across all splits.
        """
        results = []
        for i, (y_train, y_test_1s) in tqdm(
            enumerate(zip(*self.dataloader.splits)), desc="Splits", leave=False
        ):
            self.logger.debug("Initiating training on split %d", i + 1)
            self.logger.debug(
                "Number of 1s in the training set %s",
                len(y_train.data[y_train.data == 1]),
            )
            self.logger.debug(
                "Number of 0s in the training set %s",
                len(y_train.data[y_train.data == 0]),
            )

            session = smurff.MacauSession(
                **self.macau_session_kwargs,
                num_latent=num_latent,
                Ytrain=y_train,
                Ytest=y_test_1s,
                save_name=str(self.path / f"{i}:{save_name}"),
                side_info=self.side_info_loader.side_info
                if self.side_info_loader
                else None,
            )
            session.run()
            y_pred = self.predict(session)
            results.append(
                Results(y_true=self.dataloader.omim1[i].tocsr(), y_pred=y_pred)
            )
        return Evaluation(results)


def save_evaluations(results: Dict[str, Evaluation], output_path: str):
    """
    Save evaluation results to a file in binary format using pickle.

    Args:
        results (Dict[str, Evaluation]): A dictionary where keys are descriptive strings
            (e.g., latent dimensions or other identifiers) and values are `Evaluation` objects
            containing evaluation metrics and results.
        output_path (str): The file path where the results should be saved.
    """
    with open(output_path, "wb") as handler:
        pickle.dump(results, handler)
