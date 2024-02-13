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
import seaborn as sns

from autogluon_benchmark.evaluation.elo.elo_utils import compute_mle_elo, convert_results_to_battles, get_bootstrap_result


def plot_1(data, task_metadata):
    results_ranked_df = data.copy()
    results_ranked_df = pd.merge(results_ranked_df, task_metadata[['dataset', 'NumberOfInstances']], on='dataset')
    results_ranked_df['NumberOfInstancesTest'] = results_ranked_df['NumberOfInstances'] / 10
    x_name = "Inference Batch Size (NumberOfInstancesTest)"
    y_name = "Inference Speed (rows/second)"
    results_ranked_df[x_name] = results_ranked_df["NumberOfInstancesTest"]
    results_ranked_df[y_name] = results_ranked_df["rows_per_second"]
    sns.set_theme(style="darkgrid")
    # sns.color_palette("Paired")
    # mean auroc vs number of samples in the data
    g = sns.lmplot(
        data=results_ranked_df,
        x=x_name, y=y_name,
        palette='Paired',
        hue="framework",
        lowess=True,
        height=10,
        aspect=1.6,
        scatter_kws={'alpha': 0.75}
    )

    # g.set(ylim=[0, 1])
    g.set(xscale="log")
    g.set(yscale="log")
    # ax = plt.gca()
    # ax.set_title("Graph (a)")
    # plt.title('My Title')
    # Access the figure
    fig = g.fig

    # Add a title to the Figure
    fig.suptitle("AutoML Inference Throughput by Inference Batch Size", fontsize=16)
    plt.show()


def get_pareto_frontier(Xs, Ys, max_X=True, max_Y=True):
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=max_X)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if max_Y:
            if pair[1] >= pareto_front[-1][1]:
                if len(pareto_front) != 0:
                    pareto_front.append([pair[0], pareto_front[-1][1]])
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                if len(pareto_front) != 0:
                    pareto_front.append([pair[0], pareto_front[-1][1]])
                pareto_front.append(pair)
    pareto_front.append([sorted_list[-1][0], pareto_front[-1][1]])
    return pareto_front


def plot_pareto(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    title: str,
    palette='Paired',
    hue: str = "Framework",
    max_X: bool = False,
    max_Y: bool = True,
    save_path: str = None,
    show: bool = True,
):
    g = sns.relplot(
        x=x_name,
        y=y_name,
        data=data,
        palette=palette,
        hue=hue,
        height=10,
        s=300,
        # aspect=1.6,
    )

    Xs = list(data[x_name])
    Ys = list(data[y_name])
    pareto_front = get_pareto_frontier(Xs=Xs, Ys=Ys, max_X=max_X, max_Y=max_Y)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y)

    plt.ylim(None, 1)
    g.set(xscale="log")
    fig = g.fig

    plt.grid()
    # plt.gca().inverse_xaxis()

    # Add a title to the Figure
    fig.suptitle(title, fontsize=14)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_pareto_aggregated(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    data_x: pd.DataFrame = None,
    x_method: str = "median",
    y_method: str = "mean",
    max_X: bool = False,
    max_Y: bool = True,
    hue: str = "Framework",
    title: str = None,
    save_path: str = None,
    show: bool = True,
    include_method_in_axis_name: bool = True,
):
    if data_x is None:
        data_x = data
    y_vals = aggregate_stats(df=data, on=y_name, method=[y_method])[y_method]
    x_vals = aggregate_stats(df=data_x, on=x_name, method=[x_method])[x_method]
    if include_method_in_axis_name:
        x_name = f'{x_name} ({x_method})'
        y_name = f'{y_name} ({y_method})'
    df_aggregated = y_vals.to_frame(name=y_name)
    df_aggregated[x_name] = x_vals
    df_aggregated[hue] = df_aggregated.index

    plot_pareto(
        data=df_aggregated,
        x_name=x_name,
        y_name=y_name,
        title=title,
        save_path=save_path,
        max_X=max_X,
        max_Y=max_Y,
        hue=hue,
        show=show,
    )


def plot_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str = "framework",
    title: str = None,
    sort: bool = True,
    higher_is_better: bool = True,
    xlim: tuple = None,
    xscale: str = None,
    save_path: str = None,
    show: bool = True,
):
    if sort:
        order = data[[y, x]].groupby([y]).agg(["median", "mean"]).sort_values(by=[(x, "median"), (x, "mean")], ascending=not higher_is_better)
        order = list(order.index)
    else:
        order = None

    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.boxplot(
        data=data,
        ax=ax,
        y=y,
        x=x,
        order=order,
    )
    if xlim is not None:
        ax.set(xlim=xlim)
    if title is not None:
        ax.set_title(title)
    if xscale is not None:
        ax.set_xscale(xscale)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()


def aggregate_stats(df, on: str, groupby="framework", method=["mean", "median", "std"]):
    return df[[groupby, on]].groupby(groupby).agg(method)[on]


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
