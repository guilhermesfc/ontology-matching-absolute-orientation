import logging
import os
import pandas as pd
import numpy as np
from numpy.linalg import svd
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import statistics

logger = logging.getLogger(__name__)


def translate(embeddings: pd.DataFrame) -> pd.DataFrame:
    np_embeddings = embeddings.to_numpy()
    trans_vec = np_embeddings.sum(axis=0) / np_embeddings.shape[0]
    return embeddings - trans_vec


def norm_vectors(embedding: pd.DataFrame) -> pd.DataFrame:
    return embedding.div(embedding.apply(np.linalg.norm, axis=1), axis=0)


def absolute_orientation(
    embeddings: Dict["source/target", pd.DataFrame], alignments: pd.DataFrame
) -> pd.DataFrame:
    """
    calculation of Absolute Orientation (AO) following the paper of Sunipa Dev, Safia Hassan and Jeff Philips
    """
    logger.info("absolute_orientation(): Start ...")
    filtered_source_embeddings = embeddings["source"].loc[alignments.index].to_numpy()
    filtered_target_embeddings = (
        embeddings["target"].loc[alignments["target"]].to_numpy()
    )

    # should be dxd matrix - original
    outer_product = np.einsum(
        "ij,ik->jk", filtered_target_embeddings, filtered_source_embeddings
    )
    # should be dxd matrix - 1
    # outer_product = np.einsum('ij,ik->jk', filtered_source_embeddings, filtered_target_embeddings)

    U, s, VT = svd(outer_product)
    rotation_matrix = U @ VT

    source = embeddings["source"].to_numpy()
    rot_source = rotation_matrix @ source.T
    embeddings["source"] = pd.DataFrame(
        rot_source.T,
        columns=embeddings["source"].columns,
        index=embeddings["source"].index,
    )

    logger.info("absolute_orientation(): Start ... done")
    return embeddings["source"]


def Gavish_Donoho(
    resultdir: str,
    embeddings: Dict["source/target", pd.DataFrame],
    data_reduction: bool = True,
) -> pd.DataFrame:
    # https://youtu.be/epoHE2rex0g
    # https://arxiv.org/abs/1305.5870
    logger.info("Gavish_Donoho(): Start ...")
    dimension = dict()
    cleaned_embeddings = dict()
    for dataType in ["source", "target"]:

        X = embeddings[dataType].to_numpy().T
        U, S, VT = svd(X, full_matrices=0)
        N = embeddings[dataType].shape[0]
        # cutoff = (4/np.sqrt(3)) * np.sqrt(N) * sigma
        beta = X.shape[1] / X.shape[0]
        beta = beta if beta < 1 else 1 / beta
        omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
        cutoff = omega * statistics.median(S)
        dimension[dataType] = np.max(np.where(S > cutoff))

        plot_Gavish_Donoho(
            path=resultdir, N=N, S=S, r=dimension[dataType], cutoff=cutoff
        )

        if data_reduction:
            cleaned_X = (
                U[:, : (dimension[dataType] + 1)]
                @ np.diag(S[: (dimension[dataType] + 1)])
                @ VT[: (dimension[dataType] + 1), :]
            )
            cleaned_embeddings[dataType] = pd.DataFrame(
                cleaned_X.T, index=embeddings[dataType].index
            )

    if data_reduction:
        return cleaned_embeddings
    return embeddings


def plot_Gavish_Donoho(path, N, S, r, cutoff):
    ## Plot Singular Values
    cdS = np.cumsum(S) / np.sum(S)  # Cumulative energy
    r90 = np.min(np.where(cdS > 0.90))  # Find r to capture 90% energy

    fig1, ax1 = plt.subplots(1)

    ax1.semilogy(S, "-o", color="k", linewidth=2)
    ax1.semilogy(np.diag(S[: (r + 1)]), "o", color="r", linewidth=2)
    ax1.plot(
        np.array([-20, N + 20]),
        np.array([cutoff, cutoff]),
        "--",
        color="r",
        linewidth=2,
    )
    rect = patches.Rectangle(
        (-5, 20), 100, 200, linewidth=2, linestyle="--", facecolor="none", edgecolor="k"
    )
    ax1.add_patch(rect)
    plt.xlim((-10, 610))
    plt.ylim((0.003, 300))
    ax1.grid()
    plt.show()
    plt.savefig(os.path.join(path, "SingularValues.png"))

    fig2, ax2 = plt.subplots(1)

    ax2.semilogy(S, "-o", color="k", linewidth=2)
    ax2.semilogy(np.diag(S[: (r + 1)]), "o", color="r", linewidth=2)
    ax2.plot(
        np.array([-20, N + 20]),
        np.array([cutoff, cutoff]),
        "--",
        color="r",
        linewidth=2,
    )
    plt.xlim((-5, 100))
    plt.ylim((20, 200))
    ax2.grid()
    plt.show()
    plt.savefig(os.path.join(path, "SingularValues_Zoom.png"))

    fig3, ax3 = plt.subplots(1)
    ax3.plot(cdS, "-o", color="k", linewidth=2)
    ax3.plot(cdS[: (r90 + 1)], "o", color="b", linewidth=2)
    ax3.plot(cdS[: (r + 1)], "o", color="r", linewidth=2)
    plt.xticks(np.array([0, 300, r90, 600]))
    plt.yticks(np.array([0, 0.5, 0.9, 1]))
    plt.xlim((-10, 610))
    ax3.plot(
        np.array([r90, r90, -10]), np.array([0, 0.9, 0.9]), "--", color="b", linewidth=2
    )

    ax3.grid()
    plt.show()
    plt.savefig(os.path.join(path, "SingularValues_Integration.png"))
