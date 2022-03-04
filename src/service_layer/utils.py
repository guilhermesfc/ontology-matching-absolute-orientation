import logging
import pandas as pd
import numpy as np

from typing import List, Tuple
import os
import logging
import random


def split_alignments(
    alignments: pd.DataFrame, train_percentage: float = 0.8
) -> (pd.DataFrame, pd.DataFrame):
    logging.info(
        "split_alignments(alignments={}, train_percentage={})...".format(
            "df_name", train_percentage
        )
    )
    if 0 < train_percentage and train_percentage < 1:
        msk = np.random.rand(len(alignments)) < train_percentage
        train = alignments[msk]
        test = alignments[~msk]
        return train, test
    else:
        return pd.DataFrame, None


def add_noise_2_alignment(alignments: pd.DataFrame, percentage: float) -> pd.DataFrame:

    if 0 < percentage and percentage < 1:
        msk = np.random.rand(len(alignments)) < percentage
        add_noise = alignments[msk]
        keep_correct = alignments[~msk]
        tmp_list = list(add_noise["target"])
        random.shuffle(tmp_list)
        add_noise = add_noise.assign(target=tmp_list)
        return keep_correct.append(add_noise)
    elif percentage == 0:
        return alignments
    else:
        raise Exception(
            "add_noise_2_alignment: percentage has to be 0<x<1. It is :{}".format(
                percentage
            )
        )

# analyzer helper functions

def _get_condition_n(df_in, get_true=True, threshold=0., top_n=1):
    true_target  = (df_in.target == df_in['prediction_target_{}'.format(top_n)])
    false_target = (df_in.target != df_in['prediction_target_{}'.format(top_n)])
    matched      = (df_in['metric_{}'.format(top_n)] >= threshold)

    if get_true:
        return (true_target & matched)
    else:
        return (false_target & matched)


# Top N Matches returned
def _get_match(df_in, get_true=True, threshold=0., top_n=1):
    # get_true = True: True Positives
    # get_true = False: False Positives

    conditions = []

    for i in range(1, top_n+1):
        conditions.append(_get_condition_n(df_in, get_true=get_true, threshold=threshold, top_n=i))

    if top_n == 1:   cond = conditions[0]
    elif top_n == 2: cond = conditions[0] | conditions[1]
    elif top_n == 3: cond = conditions[0] | conditions[1] | conditions[2]
    elif top_n == 4: cond = conditions[0] | conditions[1] | conditions[2] | conditions[3]
    elif top_n == 5: cond = conditions[0] | conditions[1] | conditions[2] | conditions[3] | conditions[4]
    elif top_n == 6: cond = conditions[0] | conditions[1] | conditions[2] | conditions[3] | conditions[4] | conditions[5]
    elif top_n == 7: cond = conditions[0] | conditions[1] | conditions[2] | conditions[3] | conditions[4] | conditions[5] | conditions[6]
    elif top_n == 8: cond = conditions[0] | conditions[1] | conditions[2] | conditions[3] | conditions[4] | conditions[5] | conditions[6] | conditions[7]
    elif top_n == 9: cond = conditions[0] | conditions[1] | conditions[2] | conditions[3] | conditions[4] | conditions[5] | conditions[6] | conditions[7] | conditions[8]
    elif top_n ==10: cond = conditions[0] | conditions[1] | conditions[2] | conditions[3] | conditions[4] | conditions[5] | conditions[6] | conditions[7] | conditions[8] | conditions[9]

    else: return 0

    return df_in.target[ cond ].values


def compute_precision_recall_f1score(
    df_in: pd.DataFrame, threshold: float, top_n: int
) -> Tuple[float, float]:
    # precision = true positives returned / all returned = TP / (TP + FP)
    # recall = true positives returned / len(reference alignment) != TP / (TP + FN)
    true_pos = _get_match(df_in, get_true=True, threshold=threshold, top_n=top_n)
    false_pos = _get_match(df_in, get_true=False, threshold=threshold, top_n=top_n)

    precision = 0
    recall = 0
    f1_score = 0

    if (len(true_pos) + len(false_pos)) > 0:
        precision = float(len(true_pos)) / (len(true_pos) + len(false_pos))
    if len(df_in) > 0:
        recall = float(len(true_pos)) / len(df_in)
    if (recall + precision) > 0:
        f1_score = 2*(recall * precision) / (recall + precision)

    return np.round(100.0 * precision, 2), np.round(100.0 * recall, 2), np.round(100.0 * f1_score, 2)
