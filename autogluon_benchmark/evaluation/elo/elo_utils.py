from collections import defaultdict
from typing import List

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

MODEL_A = "framework_1"
MODEL_B = "framework_2"


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


def get_rank_confidence(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values(by=["Arena Elo"], ascending=False)

    elo_ratings = df["Arena Elo"].to_list()
    uppers = df["upper"].to_list()
    lowers = df["lower"].to_list()

    ranks = []

    cur_rank = 0
    prev_lower = None
    num_models = len(elo_ratings)
    for i in range(num_models):
        cur_elo = elo_ratings[i]
        cur_upper = uppers[i]
        cur_lower = lowers[i]
        if prev_lower is None or cur_upper < prev_lower:
            cur_rank = i + 1
            prev_lower = cur_lower
        ranks.append(cur_rank)

    df["Rank"] = ranks

    return df


def predict_win_rate(elo_ratings: dict, SCALE=400, BASE=10):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def _fix_missing(df, missing_A, missing_B):
    df = df.copy()
    for b in missing_B:
        df[b] = 0
    df = df.T
    for a in missing_A:
        df[a] = 0
    df = df.T
    return df


def compute_pairwise_win_fraction(battles, max_num_models=30) -> pd.DataFrame:
    unique_A = list(battles[MODEL_A].unique())
    unique_B = list(battles[MODEL_B].unique())
    missing_A = [b for b in unique_B if b not in unique_A]
    missing_B = [a for a in unique_A if a not in unique_B]
    unique_all = unique_A + missing_A
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles['winner'] == "1"],
        index=MODEL_A, columns=MODEL_B, aggfunc="size", fill_value=0)

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles['winner'] == "2"],
        index=MODEL_A, columns=MODEL_B, aggfunc="size", fill_value=0)

    # Table counting times each model wins as Model B
    tie_ptbl = pd.pivot_table(
        battles[battles['winner'] == "tie"],
        index=MODEL_A, columns=MODEL_B, aggfunc="size", fill_value=0)
    tie_ptbl *= 0.5

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(battles,
        index=MODEL_A, columns=MODEL_B, aggfunc="size", fill_value=0)

    a_win_ptbl = _fix_missing(df=a_win_ptbl, missing_A=missing_A, missing_B=missing_B)
    b_win_ptbl = _fix_missing(df=b_win_ptbl, missing_A=missing_A, missing_B=missing_B)
    tie_missing_A = [a for a in unique_all if a not in tie_ptbl.index]
    tie_missing_B = [b for b in unique_all if b not in tie_ptbl.columns]
    tie_ptbl = _fix_missing(df=tie_ptbl, missing_A=tie_missing_A, missing_B=tie_missing_B)
    num_battles_ptbl = _fix_missing(df=num_battles_ptbl, missing_A=missing_A, missing_B=missing_B)

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (
        (a_win_ptbl + b_win_ptbl.T + tie_ptbl + tie_ptbl.T) /
        (num_battles_ptbl + num_battles_ptbl.T)
    )

    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    prop_wins = prop_wins[:max_num_models]
    model_names = list(prop_wins.keys())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col


def get_arena_leaderboard(bootstrap_elo_lu: pd.DataFrame, results_df: pd.DataFrame):
    bars = pd.DataFrame(dict(
        lower=bootstrap_elo_lu.quantile(.025),
        rating=bootstrap_elo_lu.quantile(.5),
        upper=bootstrap_elo_lu.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    battles = convert_results_to_battles(results_df=results_df)
    from collections import defaultdict
    framework_battle_counts = defaultdict(int)
    framework_win_counts = defaultdict(float)
    for f, win_val in [(MODEL_A, "1"), (MODEL_B, "2")]:
        counts = battles[f].value_counts().to_dict()
        win_counts = battles[[f, "winner"]].value_counts().reset_index()
        win_counts.loc[win_counts["winner"] == "tie", "count"] *= 0.5
        win_counts = win_counts.loc[win_counts["winner"].isin([win_val, "tie"]), :]
        win_counts = win_counts.drop(columns=["winner"]).groupby(f)["count"].sum().to_dict()
        for framework in counts:
            framework_battle_counts[framework] += counts[framework]
        for framework in win_counts:
            framework_win_counts[framework] += win_counts[framework]
    framework_battle_counts = dict(framework_battle_counts)

    def _get_95_ci(upper, lower):
        return f"+{upper:.0f}/-{lower:.0f}"

    leaderboard = bars.copy()
    leaderboard["95% CI"] = [_get_95_ci(upper, lower) for upper, lower in zip(leaderboard["error_y"], leaderboard["error_y_minus"])]
    leaderboard["Arena Elo"] = np.round(leaderboard['rating'], 0).astype(int)
    leaderboard["Battles"] = leaderboard["model"].map(framework_battle_counts)
    leaderboard["Wins"] = np.round(leaderboard["model"].map(framework_win_counts), decimals=0).astype(int)
    leaderboard["Winrate"] = np.round(leaderboard["Wins"] / leaderboard["Battles"], decimals=2)
    leaderboard["Rank (Simple)"] = leaderboard["Arena Elo"].rank(method="min", ascending=False).astype(int)
    leaderboard["Model"] = leaderboard["model"]
    leaderboard = get_rank_confidence(df=leaderboard)

    results_mean_agg = results_df[["framework", "rank", "bestdiff", "loss_rescaled"]].groupby("framework").mean()
    leaderboard["mean_rank"] = leaderboard["model"].map(results_mean_agg["rank"])
    leaderboard["mean_bestdiff"] = leaderboard["model"].map(results_mean_agg["bestdiff"])
    leaderboard["mean_loss_rescaled"] = leaderboard["model"].map(results_mean_agg["loss_rescaled"])

    leaderboard["Rank Avg"] = np.round(leaderboard["mean_rank"], decimals=1)
    leaderboard["Champ Delta %"] = np.round(leaderboard["mean_bestdiff"] * 100, decimals=1)
    leaderboard["Rescaled Loss"] = np.round(leaderboard["mean_loss_rescaled"], decimals=2)

    leaderboard_print = leaderboard[[
        "Rank",
        "Model",
        "Arena Elo",
        "95% CI",
        # "Battles",
        # "Wins",
        "Winrate",
        "Rescaled Loss",
        # "Rank Avg",
        "Champ Delta %",
    ]]

    return leaderboard, leaderboard_print
