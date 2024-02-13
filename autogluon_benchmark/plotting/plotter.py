import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from .plot_pareto_frontier import plot_pareto_aggregated
from .plot_boxplot import plot_boxplot
from ..evaluation.elo.elo_utils import compute_elo_ratings


def visualize_bootstrap_scores(df: pd.DataFrame, title: str):
    bars = pd.DataFrame(dict(
        lower=df.quantile(.025),
        rating=df.quantile(.5),
        upper=df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating",
                      height=1000, width=1600)
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
        self.results_ranked_fillna_df = results_ranked_fillna_df
        self.results_ranked_df = results_ranked_df
        self._verify_integrity()
        self.save_dir = save_dir
        self.show = show

    def _filename(self, name):
        if self.save_dir is not None:
            return os.path.join(self.save_dir, name)
        else:
            return None

    def _verify_integrity(self):
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
        fig = visualize_bootstrap_scores(bootstrap_elo_lu, "Bootstrap of MLE Elo Rating Estimates")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
        if self.show:
            fig.show()

    def plot_all(
        self,
        calibration_framework: str = None,
        calibration_elo: float = None,
    ):
        self.plot_boxplot_rescaled_accuracy()
        self.plot_boxplot_champion_loss_delta()
        self.plot_boxplot_rank()
        self.plot_boxplot_time_train()
        self.plot_boxplot_samples_per_second()
        self.plot_pareto_time_infer()
        self.plot_pareto_time_train()
        self.plot_critical_difference()
        self.plot_elo_ratings(
            calibration_framework=calibration_framework,
            calibration_elo=calibration_elo,
        )
