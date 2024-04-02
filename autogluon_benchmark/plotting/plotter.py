import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from autogluon.common.savers import save_pd, save_str

from .plot_pareto_frontier import plot_pareto_aggregated
from .plot_boxplot import plot_boxplot
from ..evaluation.elo.elo_utils import compute_elo_ratings, convert_results_to_battles

MODEL_A = "framework_1"
MODEL_B = "framework_2"


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

    print(leaderboard)
    print(leaderboard_print)

    return leaderboard, leaderboard_print


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    from collections import defaultdict
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


def plot_winrate_expectation_by_elo(bootstrap_elo_lu):
    win_rate = predict_win_rate(dict(bootstrap_elo_lu.quantile(0.5)))
    ordered_models = win_rate.mean(axis=1).sort_values(ascending=False).index
    ordered_models = ordered_models[:30]
    fig = px.imshow(win_rate.loc[ordered_models, ordered_models],
                    color_continuous_scale='RdBu', text_auto=".2f",
                    title="Predicted Win Rate Using Elo Ratings for Model A in an A vs. B Battle")
    fig.update_layout(xaxis_title="Model B",
                      yaxis_title="Model A",
                      xaxis_side="top", height=900, width=900,
                      title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                      "Model A: %{y}<br>Model B: %{x}<br>Win Rate: %{z}<extra></extra>")
    return fig


def _fix_missing(df, missing_A, missing_B):
    df = df.copy()
    for b in missing_B:
        df[b] = 0
    df = df.T
    for a in missing_A:
        df[a] = 0
    df = df.T
    return df


def compute_pairwise_win_fraction(battles, max_num_models=30):
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


def visualize_pairwise_win_fraction(battles, title, max_num_models=30):
    row_beats_col = compute_pairwise_win_fraction(battles, max_num_models)
    fig = px.imshow(row_beats_col, color_continuous_scale='RdBu',
                    text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B: Loser",
                  yaxis_title="Model A: Winner",
                  xaxis_side="top", height=900, width=900,
                  title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                  "Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>")

    return fig


def visualize_bootstrap_scores(df: pd.DataFrame, title: str):
    bars = pd.DataFrame(dict(
        lower=df.quantile(.025),
        rating=df.quantile(.5),
        upper=df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 0).astype(int)
    num_models = len(bars["model"].unique())
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating",
                      height=600, width=200 + num_models*40)
    return fig


class Plotter:
    """
    Class that can produce plots from AutoMLBenchmark results files
    """
    def __init__(
        self,
        results_ranked_fillna_df: pd.DataFrame,
        results_ranked_df: pd.DataFrame,
        save_dir: str = None,
        show: bool = True,
    ):
        self.results_ranked_fillna_df = results_ranked_fillna_df.copy()
        self.results_ranked_df = results_ranked_df.copy()
        if "fold" in self.results_ranked_fillna_df:
            self.results_ranked_fillna_df["dataset"] = self.results_ranked_fillna_df["dataset"] + "_" + self.results_ranked_fillna_df["fold"].astype(str)
            self.results_ranked_df["dataset"] = self.results_ranked_df["dataset"] + "_" + self.results_ranked_df["fold"].astype(str)
        self._verify_integrity()
        self.save_dir = save_dir
        self.show = show

    def _filename(self, name):
        if self.save_dir is not None:
            return os.path.join(self.save_dir, name)
        else:
            return None

    def _verify_integrity(self):
        expected_columns = [
            "framework",
            "dataset",
            "metric_error",
            "bestdiff",
            "loss_rescaled",
            "rank",
            "time_train_s",
            "time_infer_s",
        ]
        for column in expected_columns:
            assert column in self.results_ranked_df
            assert column in self.results_ranked_fillna_df

        unique_frameworks = set(self.results_ranked_fillna_df["framework"].unique())
        unique_datasets = set(self.results_ranked_fillna_df["dataset"].unique())
        assert unique_frameworks == set(self.results_ranked_df["framework"].unique())
        assert len(self.results_ranked_fillna_df) == (len(unique_frameworks) * len(unique_datasets))
        assert len(self.results_ranked_df.drop_duplicates(subset=["framework", "dataset"])) == len(self.results_ranked_df)
        assert len(self.results_ranked_fillna_df.drop_duplicates(subset=["framework", "dataset"])) == len(self.results_ranked_fillna_df)

    def plot_boxplot_rank(self):
        save_path = self._filename("boxplot_rank.png")
        x = "Rank, Lower is Better"
        data = self.results_ranked_fillna_df[["framework", "rank"]].copy()
        data[x] = data["rank"]
        max_rank = data[x].max()
        plot_boxplot(
            data=data,
            x=x,
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=False,
            xlim=(0.995, max_rank),
            save_path=save_path,
            show=self.show,
        )

    def plot_boxplot_rescaled_accuracy(self):
        save_path = self._filename("boxplot_rescaled_accuracy.png")
        x = "Rescaled Accuracy, Higher is Better"
        data = self.results_ranked_fillna_df[["framework", "loss_rescaled"]].copy()
        data[x] = 1 - data["loss_rescaled"]
        plot_boxplot(
            data=data,
            x=x,
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=True,
            xlim=(-0.001, 1),
            save_path=save_path,
            show=self.show,
        )

    def plot_boxplot_time_train(self):
        save_path = self._filename("boxplot_time_train.png")
        x = "Train Time (seconds), Lower is Better"
        data = self.results_ranked_df[["framework", "time_train_s"]].copy()
        data[x] = data["time_train_s"]
        plot_boxplot(
            data=data,
            x=x,
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=False,
            xscale="log",
            # xlim=(-0.001, 1),
            save_path=save_path,
            show=self.show,
        )

    def plot_boxplot_samples_per_second(self):
        save_path = self._filename("boxplot_samples_per_second.png")
        x = "Samples per Second (Inference), Higher is Better"
        data = self.results_ranked_df[["framework", "time_infer_s"]].copy()
        data[x] = 1/data["time_infer_s"]
        plot_boxplot(
            data=data,
            x=x,
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=True,
            xscale="log",
            # xlim=(-0.001, 1),
            save_path=save_path,
            show=self.show,
        )

    def plot_boxplot_samples_per_dollar(self, seconds_per_dollar: float):
        dollars_per_hour = 3600 / seconds_per_dollar
        save_path = self._filename("boxplot_samples_per_dollar.png")
        x = f"Samples per Dollar (Inference), Higher is Better (based on ${dollars_per_hour:.3f}/hour of compute)"
        data = self.results_ranked_df[["framework", "time_infer_s"]].copy()

        data["Samples per Second (Inference), Higher is Better"] = 1/data["time_infer_s"]
        data[x] = data["Samples per Second (Inference), Higher is Better"] * seconds_per_dollar
        plot_boxplot(
            data=data,
            x=x,
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=True,
            xscale="log",
            # xlim=(-0.001, 1),
            save_path=save_path,
            show=self.show,
        )

    def plot_boxplot_champion_loss_delta(self):
        save_path = self._filename("boxplot_champion_loss_delta.png")
        x = "Champion Loss Delta (%), Lower is Better"
        data = self.results_ranked_fillna_df[["framework", "bestdiff"]].copy()
        data[x] = data["bestdiff"] * 100
        plot_boxplot(
            data=data,
            x=x,
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=False,
            xlim=(-0.1, 100),
            save_path=save_path,
            show=self.show,
        )

    def plot_pareto_time_infer(self):
        save_path = self._filename("pareto_front_time_infer.png")
        y_name = "Rescaled Accuracy"
        x_name = "Inference Time Per-Row (seconds)"
        title = f"AutoMLBenchmark 2023 Accuracy vs Inference Time (104 datasets, 10-fold)"
        data_x = self.results_ranked_df.copy()
        data_x[x_name] = data_x["time_infer_s"]
        data = self.results_ranked_fillna_df.copy()
        data[y_name] = 1 - data["loss_rescaled"]
        plot_pareto_aggregated(
            data=data,
            data_x=data_x,
            x_name=x_name,
            y_name=y_name,
            x_method="median",
            y_method="mean",
            max_X=False,
            max_Y=True,
            title=title,
            save_path=save_path,
            show=self.show,
        )

    def plot_pareto_time_train(self):
        save_path = self._filename("pareto_front_time_train.png")
        y_name = "Rescaled Accuracy"
        x_name = "Train Time (seconds)"
        title = f"AutoMLBenchmark 2023 Accuracy vs Train Time (104 datasets, 10-fold)"
        data_x = self.results_ranked_df.copy()
        data_x[x_name] = data_x["time_train_s"]
        data = self.results_ranked_fillna_df.copy()
        data[y_name] = 1 - data["loss_rescaled"]
        plot_pareto_aggregated(
            data=data,
            data_x=data_x,
            x_name=x_name,
            y_name=y_name,
            x_method="mean",
            y_method="mean",
            max_X=False,
            max_Y=True,
            title=title,
            save_path=save_path,
            show=self.show,
        )

    def plot_critical_difference(self):
        save_path = self._filename("critical_difference.png")
        from autorank import autorank, plot_stats
        fig, axes = plt.subplots(1, 1, figsize=(20, 3))
        data = self.results_ranked_fillna_df.pivot_table(index="dataset", columns="framework", values="metric_error")
        result = autorank(data, alpha=0.05, verbose=False, order="ascending", force_mode="nonparametric")
        plot_stats(result, ax=axes, width=8, allow_insignificant=True)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        if self.show:
            plt.show()

    def plot_elo_ratings(
        self,
        seed: int = 0,
        calibration_framework: str = None,
        calibration_elo: float = None,
        INIT_RATING: float = 1000,
        BOOTSTRAP_ROUNDS: int = 100,
        SCALE: int = 400,
    ):
        save_path = self._filename("elo_ratings.png")
        bootstrap_elo_lu = compute_elo_ratings(
            results_ranked_fillna_df=self.results_ranked_fillna_df,
            seed=seed,
            calibration_framework=calibration_framework,
            calibration_elo=calibration_elo,
            INIT_RATING=INIT_RATING,
            BOOTSTRAP_ROUNDS=BOOTSTRAP_ROUNDS,
            SCALE=SCALE,
        )
        fig = visualize_bootstrap_scores(bootstrap_elo_lu, "Elo Confidence Intervals on Model Strength (via Bootstrapping)")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
        if self.show:
            fig.show()
        self._plot_pairwise_winrate_expectation_by_elo(bootstrap_elo_lu=bootstrap_elo_lu)
        return bootstrap_elo_lu

    def _get_arena_leaderboard(self, bootstrap_elo_lu):
        save_path_markdown = self._filename("leaderboard.md")
        save_path = self._filename("leaderboard.csv")
        save_path_pretty = self._filename("leaderboard_pretty.csv")
        leaderboard, leaderboard_print = get_arena_leaderboard(bootstrap_elo_lu=bootstrap_elo_lu, results_df=self.results_ranked_fillna_df)
        leaderboard_markdown = leaderboard_print.to_markdown(index=False)
        from autogluon.common.savers import save_str
        if save_path:
            save_pd.save(path=save_path, df=leaderboard)
            save_pd.save(path=save_path_pretty, df=leaderboard_print)
            save_str.save(path=save_path_markdown, data=leaderboard_markdown)
        if self.show:
            print(leaderboard_markdown)
        return leaderboard, leaderboard_print

    def _plot_pairwise_winrate_expectation_by_elo(self, bootstrap_elo_lu):
        save_path = self._filename("pairwise_winrate_expectation_by_elo.png")
        fig = plot_winrate_expectation_by_elo(bootstrap_elo_lu=bootstrap_elo_lu)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
        if self.show:
            fig.show()

    def plot_pairwise_winrate(self):
        save_path = self._filename("pairwise_winrate.png")
        battles = convert_results_to_battles(results_df=self.results_ranked_fillna_df)
        fig = visualize_pairwise_win_fraction(battles=battles, title="Fraction of Model A Wins for All A vs. B Battles")
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
        if self.show:
            fig.show()

    def plot_average_winrate(self):
        save_path = self._filename("average_winrate.png")
        battles = convert_results_to_battles(results_df=self.results_ranked_fillna_df)
        row_beats_col_freq = compute_pairwise_win_fraction(battles=battles)
        fig = px.bar(
            row_beats_col_freq.mean(axis=1).sort_values(ascending=False),
            title="Average Win Rate Against All Other Models (Assuming Uniform Sampling)",
            text_auto=".2f",
        )
        fig.update_layout(
            yaxis_title="Average Win Rate",
            xaxis_title="Model",
            showlegend=False,
        )
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
        if self.show:
            fig.show()

    def plot_all(
        self,
        calibration_framework: str = None,
        calibration_elo: float = None,
        BOOTSTRAP_ROUNDS: int = 100,
    ):
        self.plot_boxplot_rescaled_accuracy()
        self.plot_boxplot_champion_loss_delta()
        self.plot_boxplot_rank()
        self.plot_boxplot_time_train()
        self.plot_boxplot_samples_per_second()
        self.plot_pareto_time_infer()
        self.plot_pareto_time_train()
        self.plot_critical_difference()
        self.plot_average_winrate()
        self.plot_pairwise_winrate()
        bootstrap_elo_lu = self.plot_elo_ratings(
            calibration_framework=calibration_framework,
            calibration_elo=calibration_elo,
            BOOTSTRAP_ROUNDS=BOOTSTRAP_ROUNDS,
        )
        self._get_arena_leaderboard(bootstrap_elo_lu=bootstrap_elo_lu)
