# pylint: disable=R0913
"""
Figures module
==============

This module provides post-processing functions to generate and save visualizations 
of evaluation metrics such as ROC curves and BEDROC boxplots. These visualizations 
help in comparing model performance across different configurations, splits, and metrics.

"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from NEGradient_GenePriority.evaluation.evaluation import Evaluation
from NEGradient_GenePriority.postprocessing.model_evaluation_collection import (
    ModelEvaluationCollection,
)


def plot_roc_curves(
    evaluation_collection: ModelEvaluationCollection,
    output_file: str,
    cmap: str = "Set2",
    figsize=(8, 6),
):
    """
    Plots average ROC curves for multiple models and saves the plot to a file.

    Args:
        evaluation_collection (ModelEvaluationCollection): A collection of `Evaluation`
            objects, each containing the evaluation of a model.
        output_file (str): File path where the ROC curve plot will be saved.
        cmap (str, optional): Matplotlib colormap to use for line colors. Defaults to "Set2".
        figsize (tuple, optional): Figure size in inches (width, height). Defaults to (8, 6).

    """
    colors = plt.get_cmap(cmap).colors
    fig = plt.figure(figsize=figsize)
    for i, (name, evaluation) in enumerate(evaluation_collection.items()):
        fpr_tpr_avg = evaluation.compute_roc_curve().mean(axis=0)
        fpr, tpr = fpr_tpr_avg
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            c=colors[i],
            label=name,
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random Guess")
    plt.xlabel("Average FPR")
    plt.ylabel("Average TPR")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.suptitle("Average ROC Curve across all Diseases.", fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_bedroc_boxplots(
    bedroc: np.ndarray,
    model_names: List[str],
    output_file: str,
    cmap: str = "Set2",
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plots boxplots of BEDROC scores for multiple alpha values and latent dimensions.

    Args:
        bedroc (np.ndarray): BEDROC scores array of shape (alphas, diseases, models).
            Each entry represents the BEDROC score for a specific alpha value, disease,
            and model.
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the BEDROC boxplot figure will be saved.
        cmap (str, optional): Matplotlib colormap for the boxplot colors. Defaults to "Set2".
        figsize (Tuple[int, int], optional): Figure size in inches (width, height).
            Defaults to (10, 8).

    """
    subplots_config = (2, np.ceil(len(Evaluation.alphas) / 2).astype(int))
    fig, axs = plt.subplots(*subplots_config, figsize=figsize)
    axs = axs.flatten()
    for i, alpha in enumerate(Evaluation.alphas):
        sns.boxplot(data=bedroc[i], ax=axs[i], palette=cmap)
        axs[i].set_xticks(range(bedroc.shape[2]))  # Set fixed positions for ticks
        axs[i].set_xticklabels(model_names, rotation=0, ha="center", fontsize=10)
        axs[i].set_title(
            f"{float(alpha):.1f}\nTop {Evaluation.alpha_map[alpha]}", fontsize=12
        )
        axs[i].grid(axis="y", alpha=0.3)

    # Disable unused subplots
    if len(Evaluation.alphas) % 2 == 1:
        axs[-1].axis("off")
    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
    fig.suptitle("Averaged BEDROC", fontsize=14)
    plt.savefig(output_file, dpi=300)
    plt.close()
