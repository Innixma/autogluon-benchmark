from typing import List

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def compute_mle_elo(
    df: pd.DataFrame,
    SCALE: int = 400,
    BASE: int = 10,
    INIT_RATING: int = 1000,
    calibration_framework: str = None,
    calibration_elo: float = None
) -> pd.Series:
    """
    Adapted from ChatBot Arena: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=4_x-vXL4yxvC

    Parameters
    ----------
    df
    SCALE
    BASE
    INIT_RATING
    calibration_framework
    calibration_elo

    Returns
    -------

    """
    models = pd.concat([df["framework_1"], df["framework_2"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["framework_1"]]] = +math.log(BASE)
    X[np.arange(n), models[df["framework_2"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "1"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = df["winner"] == "tie"
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    if calibration_framework is not None:
        if calibration_elo is None:
            calibration_elo = INIT_RATING
        # calibrate random forest to 800
        elo_scores += (calibration_elo-elo_scores[models[calibration_framework]])
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles: pd.DataFrame, func_compute_elo, rng, num_round: int = None, func_kwargs=None):
    rows = []
    if func_kwargs is None:
        func_kwargs = {}
    if num_round is None:
        rows.append(func_compute_elo(battles, **func_kwargs))
    else:
        num_battles = len(battles)
        for i in tqdm(range(num_round), desc="bootstrap"):
            battles_new = battles.sample(n=num_battles, replace=True, random_state=rng, axis=0)
            rows.append(func_compute_elo(battles_new, **func_kwargs))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def calc_battle_outcome(error_1: float, error_2: float) -> str:
    if error_1 < error_2:
        winner = "1"
    elif error_1 > error_2:
        winner = "2"
    else:
        winner = "tie"
    return winner


def convert_results_to_battles(
    results_df: pd.DataFrame,
    frameworks: List[str] = None,
    datasets: List[str] = None,
) -> pd.DataFrame:
    results_df = results_df[["framework", "dataset", "metric_error"]]
    if datasets is not None:
        results_df = results_df[results_df["dataset"].isin(datasets)]
    if frameworks is not None:
        results_df = results_df[results_df["framework"].isin(frameworks)]
    results_pairs_df = pd.merge(results_df, results_df, on="dataset", suffixes=('_1', '_2'))
    results_pairs_df = results_pairs_df[results_pairs_df["framework_1"] != results_pairs_df["framework_2"]]
    results_pairs_df["winner"] = [
        calc_battle_outcome(
            error_1=error_1,
            error_2=error_2,
        ) for error_1, error_2 in zip(results_pairs_df["metric_error_1"], results_pairs_df["metric_error_2"])
    ]
    return results_pairs_df[["framework_1", "framework_2", "winner", "dataset"]]
