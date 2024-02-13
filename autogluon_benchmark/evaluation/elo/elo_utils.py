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


def get_bootstrap_result(battles: pd.DataFrame, func_compute_elo, rng=None, num_round: int = None, func_kwargs=None):
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

    # Avoid counting each battle twice (dedupe A vs B with B vs A)
    frameworks_unique = list(results_pairs_df["framework_1"].unique())
    valid_framework_pairs = []
    for i in range(len(frameworks_unique)):
        f1 = frameworks_unique[i]
        for j in range(i+1, len(frameworks_unique)):
            f2 = frameworks_unique[j]
            valid_framework_pairs.append((f1, f2))
    valid_framework_pairs = set(valid_framework_pairs)
    pairs_to_keep = [
        (framework_1, framework_2) in valid_framework_pairs for framework_1, framework_2 in zip(results_pairs_df["framework_1"], results_pairs_df["framework_2"])
    ]
    results_pairs_df = results_pairs_df.iloc[pairs_to_keep]
    return results_pairs_df[["framework_1", "framework_2", "winner", "dataset"]]


def compute_elo_ratings(
    results_ranked_fillna_df: pd.DataFrame,
    seed: int = 0,
    calibration_framework=None,
    calibration_elo=None,
    INIT_RATING: float = 1000,
    BOOTSTRAP_ROUNDS: int = 100,
    SCALE: int = 400,
) -> pd.DataFrame:
    battles = convert_results_to_battles(results_df=results_ranked_fillna_df)
    rng = np.random.default_rng(seed=seed)
    bootstrap_elo_lu = get_bootstrap_result(
        battles=battles,
        func_compute_elo=compute_mle_elo,
        num_round=BOOTSTRAP_ROUNDS,
        rng=rng,
        func_kwargs={
            "INIT_RATING": INIT_RATING,
            "SCALE": SCALE,
            "calibration_framework": calibration_framework,
            "calibration_elo": calibration_elo,
        }
    )
    return bootstrap_elo_lu


def compute_elo_rating_dataset_contributon(
    results_ranked_fillna_df: pd.DataFrame,
    seed: int = 0,
    calibration_framework=None,
    calibration_elo=None,
    INIT_RATING: float = 1000,
    SCALE: int = 400,
) -> pd.DataFrame:
    datasets = list(results_ranked_fillna_df["dataset"].unique())
    battles = convert_results_to_battles(results_ranked_fillna_df, datasets=datasets)

    rng = np.random.default_rng(seed=seed)
    bootstrap_elo_lu = get_bootstrap_result(
        battles=battles,
        func_compute_elo=compute_mle_elo,
        num_round=None,
        rng=rng,
        func_kwargs={
            "INIT_RATING": INIT_RATING,
            "SCALE": SCALE,
            "calibration_framework": calibration_framework,
            "calibration_elo": calibration_elo,
        }
    )

    bars = pd.DataFrame(dict(
        rating=bootstrap_elo_lu.quantile(.5),
    )).sort_values("rating", ascending=False)

    elo_impact_by_dataset_list = []
    for dataset_to_skip in datasets:
        battles_w_dataset_removed = battles[battles["dataset"] != dataset_to_skip]
        bootstrap_elo_lu_w_dataset_removed = get_bootstrap_result(
            battles=battles_w_dataset_removed,
            func_compute_elo=compute_mle_elo,
            num_round=None,
            rng=rng,
            func_kwargs={
                "INIT_RATING": INIT_RATING,
                "SCALE": SCALE,
                "calibration_framework": calibration_framework,
                "calibration_elo": calibration_elo,
            }
        )
        bars_by_dataset = pd.DataFrame(dict(
            rating=bootstrap_elo_lu_w_dataset_removed.quantile(.5),
        ))

        delta = bars["rating"] - bars_by_dataset["rating"]
        delta.name = dataset_to_skip
        elo_impact_by_dataset_list.append(delta)
    elo_impact_by_dataset = pd.concat(elo_impact_by_dataset_list, axis=1)
    return elo_impact_by_dataset
