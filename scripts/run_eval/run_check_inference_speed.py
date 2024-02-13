"""
TODO: Refactor this, it is currently very hacky
"""

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import os

from autogluon.common.loaders import load_pd

import matplotlib.pyplot as plt

from autogluon_benchmark.evaluation.elo.elo_utils import compute_mle_elo, convert_results_to_battles, get_bootstrap_result
from autogluon_benchmark.plotting.plot_pareto_frontier import plot_pareto_aggregated
from autogluon_benchmark.plotting.plot_boxplot import plot_boxplot


def visualize_bootstrap_scores(df, title):
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


def compute_elo_ratings(
    results_ranked_fillna_df: pd.DataFrame,
    seed: int = 0,
    calibration_framework=None,
    calibration_elo=None,
    INIT_RATING: float = 1000,
    BOOTSTRAP_ROUNDS: int = 100,
    SCALE: int = 400,
    save_path: str = None,
    show: bool = True,
):
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

    fig = visualize_bootstrap_scores(bootstrap_elo_lu, "Bootstrap of MLE Elo Rating Estimates")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
    if show:
        fig.show()


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
        # path_figures = f"{benchmark_evaluator.results_dir_output}figures/"
        # Path(path_figures).mkdir(parents=True, exist_ok=True)
        # plt.savefig(f"{path_figures}ranks.png")
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        if self.show:
            plt.show()

    def plot_elo_ratings(
        self,
        seed: int = 0,
        calibration_framework=None,
        calibration_elo=None,
        INIT_RATING: float = 1000,
        BOOTSTRAP_ROUNDS: int = 100,
        SCALE: int = 400,
    ):
        save_path = self._filename("elo_ratings.png")
        compute_elo_ratings(
            results_ranked_fillna_df=self.results_ranked_fillna_df,
            seed=seed,
            calibration_framework=calibration_framework,
            calibration_elo=calibration_elo,
            INIT_RATING=INIT_RATING,
            BOOTSTRAP_ROUNDS=BOOTSTRAP_ROUNDS,
            SCALE=SCALE,
            save_path=save_path,
            show=self.show,
        )


def compute_elo_ratings_dataset_contribution(
    results_ranked_fillna_df: pd.DataFrame,
    seed: int = 0,
    calibration_framework=None,
    calibration_elo=None,
    INIT_RATING: float = 1000,
    BOOTSTRAP_ROUNDS: int = None,
    SCALE: int = 400,
    save_path: str = None,
    show: bool = True,
):
    datasets = list(results_ranked_fillna_df["dataset"].unique())

    elo_gaps = []
    n_datasets = len(datasets)

    for i in range(n_datasets):
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
            lower=bootstrap_elo_lu.quantile(.025),
            rating=bootstrap_elo_lu.quantile(.5),
            upper=bootstrap_elo_lu.quantile(.975))).sort_values("rating", ascending=False)

        framework_elo_to_track = "AutoGluon 1.0 (Best, 4h8c)"
        elo_to_track = bars.loc[framework_elo_to_track]["rating"]
        bars_without_elo_to_track = bars[bars.index != framework_elo_to_track]
        max_rating_without_track = bars_without_elo_to_track["rating"].max()
        elo_gap = elo_to_track - max_rating_without_track
        elo_gaps.append(elo_gap)
        print(f"{elo_gaps[-1]} | {elo_gaps}")

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

        best_dataset = elo_impact_by_dataset.T[framework_elo_to_track].idxmax()
        datasets = [d for d in datasets if d != best_dataset]
        print(f"REMOVING BEST DATASET: {best_dataset} | {len(datasets)} remain...")

    fig = visualize_bootstrap_scores(bootstrap_elo_lu, "Bootstrap of MLE Elo Rating Estimates")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
    if show:
        fig.show()


# TODO: Fix the input so that everything is in a function and it can be called at the end of `run_eval_autogluon_v1.py`
if __name__ == '__main__':
    # results_dir = "s3://autogluon-zeroshot/autogluon_v1/"  # s3 path
    results_dir = 'data/results/output/openml/autogluon_v1/'  # local path
    results_dir_input = results_dir

    # Price Spot Instance m5.2xlarge (US East Ohio) on May 8th 2022 : $0.0873 / hour
    price_per_hour = 0.0873
    hour_per_dollar = 1 / price_per_hour
    seconds_per_dollar = hour_per_dollar * 3600

    plotter_root_dir = "test_out3/"

    for problem_type in [
        "all",
        "binary",
        "multiclass",
        "regression",
    ]:
        run_path_prefix = f'4h8c/{problem_type}/'
        run_path_prefix_fillna = f'4h8c_fillna/{problem_type}/'
        results_ranked_df = load_pd.load(f'{results_dir_input}{run_path_prefix}results_ranked_by_dataset_valid.csv')
        results_ranked_fillna_df = load_pd.load(f'{results_dir_input}{run_path_prefix_fillna}results_ranked_by_dataset_valid.csv')

        results_ranked_df['dataset'] = results_ranked_df['dataset'].astype(str)
        results_ranked_fillna_df['dataset'] = results_ranked_fillna_df['dataset'].astype(str)

        if problem_type != "all":
            results_ranked_df = results_ranked_df[results_ranked_df["problem_type"] == problem_type]
            results_ranked_fillna_df = results_ranked_fillna_df[results_ranked_fillna_df["problem_type"] == problem_type]

        plotter = Plotter(
            results_ranked_fillna_df=results_ranked_fillna_df,
            results_ranked_df=results_ranked_df,
            save_dir=f"{plotter_root_dir}{problem_type}/",
            show=True,
        )

        plotter.plot_boxplot_rescaled_accuracy()
        plotter.plot_boxplot_champion_loss_delta()
        plotter.plot_boxplot_rank()
        plotter.plot_boxplot_time_train()
        plotter.plot_boxplot_samples_per_second()
        plotter.plot_boxplot_samples_per_dollar(seconds_per_dollar=seconds_per_dollar)
        plotter.plot_pareto_time_infer()
        plotter.plot_pareto_time_train()
        plotter.plot_critical_difference()
        plotter.plot_elo_ratings(
            calibration_framework="RandomForest (2023, 4h8c)",
            calibration_elo=800,
        )

    # from autogluon.common.utils.s3_utils import upload_s3_folder, upload_file
    # upload_s3_folder(bucket="autogluon-zeroshot", prefix="autogluon_v1/", folder_to_upload=plotter_root_dir)
    # import shutil
    # shutil.make_archive("results", 'zip', plotter_root_dir)
    # upload_file(file_name="results.zip", bucket="autogluon-zeroshot", prefix="autogluon_v1")
