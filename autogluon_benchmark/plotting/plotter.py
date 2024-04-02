import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from autogluon.common.savers import save_pd

from .plot_pareto_frontier import plot_pareto_aggregated
from .plot_boxplot import plot_boxplot
from .plotter_utils import (
    compute_pairwise_win_fraction,
    plot_winrate_expectation_by_elo,
    visualize_bootstrap_scores,
    visualize_pairwise_win_fraction,
)
from ..evaluation.elo.elo_utils import compute_elo_ratings, convert_results_to_battles, get_arena_leaderboard


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
