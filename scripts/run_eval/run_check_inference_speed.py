"""
TODO: Refactor this, it is currently very hacky
"""


import numpy as np
import pandas as pd

from autogluon.common.loaders import load_pd

import matplotlib.pyplot as plt
import seaborn as sns


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
    save_prefix: str = None,
    palette='Paired',
    hue: str = "Framework",
    max_X: bool = False,
    max_Y: bool = True,
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
    if save_prefix is not None:
        if problem_type is not None:
            save_path = f"{save_prefix}pareto_front_{problem_type}.png"
        else:
            save_path = f"{save_prefix}pareto_front.png"

        plt.savefig(save_path)
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
    save_prefix: str = None,
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
        save_prefix=save_prefix,
        max_X=max_X,
        max_Y=max_Y,
        hue=hue,
    )


# TODO: Save plots
def plot_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str = "framework",
    title: str = None,
    sort: bool = True,
    higher_is_better: bool = True,
    xlim: tuple = None,
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

    plt.show()


def aggregate_stats(df, on: str, groupby="framework", method=["mean", "median", "std"]):
    return df[[groupby, on]].groupby(groupby).agg(method)[on]


class Plotter:
    def __init__(
        self,
        results_ranked_fillna_df: pd.DataFrame,
        results_ranked_df: pd.DataFrame,
        results_ranked_overall_df: pd.DataFrame,
    ):
        self.results_ranked_fillna_df = results_ranked_fillna_df
        self.results_ranked_df = results_ranked_df
        self.results_ranked_overall_df = results_ranked_overall_df
        self._verify_integrity()

    def _verify_integrity(self):
        unique_frameworks = set(self.results_ranked_fillna_df["framework"].unique())
        unique_datasets = set(self.results_ranked_fillna_df["dataset"].unique())
        assert unique_frameworks == set(self.results_ranked_df["framework"].unique())
        assert unique_frameworks == set(self.results_ranked_overall_df["framework"].unique())
        assert len(self.results_ranked_overall_df) == len(unique_frameworks)
        assert len(results_ranked_fillna_df) == (len(unique_frameworks) * len(unique_datasets))

    def plot_boxplot_rank(self):
        data = self.results_ranked_fillna_df[["framework", "rank"]].copy()
        data["Rank, Lower is Better"] = data["rank"]
        max_rank = data["Rank, Lower is Better"].max()
        plot_boxplot(
            data=data,
            x="Rank, Lower is Better",
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=False,
            xlim=(0.995, max_rank),
        )

    def plot_boxplot_rescaled_accuracy(self):
        data = self.results_ranked_fillna_df[["framework", "loss_rescaled"]].copy()
        data["Rescaled Accuracy, Higher is Better"] = 1 - data["loss_rescaled"]
        plot_boxplot(
            data=data,
            x="Rescaled Accuracy, Higher is Better",
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=True,
            xlim=(-0.001, 1),
        )

    def plot_boxplot_champion_loss_delta(self):
        data = self.results_ranked_fillna_df[["framework", "bestdiff"]].copy()
        data["Champion Loss Delta (%), Lower is Better"] = data["bestdiff"] * 100
        plot_boxplot(
            data=data,
            x="Champion Loss Delta (%), Lower is Better",
            y="framework",
            title="AutoMLBenchmark 2023 Results (104 datasets, 10 folds)",
            sort=True,
            higher_is_better=False,
            xlim=(-0.1, 100),
        )

    def plot_pareto_infer_time(self, save_prefix: str):
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
            save_prefix=save_prefix,
        )

    def plot_pareto_train_time(self, save_prefix: str):
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
            save_prefix=save_prefix,
        )


# TODO: Fix the input so that everything is in a function and it can be called at the end of `run_eval_autogluon_v1.py`
if __name__ == '__main__':
    # results_dir = "s3://autogluon-zeroshot/autogluon_v1/"  # s3 path
    results_dir = 'data/results/output/openml/autogluon_v1/'  # local path
    results_dir_input = results_dir
    results_dir_output = results_dir
    problem_type = 'all'
    for problem_type in [
        "all",
        # "binary",
        # "multiclass",
        # "regression",
    ]:
        run_path_prefix = f'4h8c/{problem_type}/'
        run_path_prefix_overall = f'4h8c_fillna/{problem_type}/'

        results_ranked_df = load_pd.load(f'{results_dir_input}{run_path_prefix}results_ranked_by_dataset_valid.csv')
        results_ranked_overall_df = load_pd.load(f'{results_dir_input}{run_path_prefix_overall}results_ranked_valid.csv')
        results_ranked_fillna_df = load_pd.load(f'{results_dir_input}{run_path_prefix_overall}results_ranked_by_dataset_valid.csv')

        results_ranked_df['dataset'] = results_ranked_df['dataset'].astype(str)
        results_ranked_fillna_df['dataset'] = results_ranked_fillna_df['dataset'].astype(str)
        results_ranked_df['rows_per_second'] = 1 / results_ranked_df['time_infer_s']

        frameworks = list(results_ranked_df['framework'].unique())
        print(frameworks)

        # Price Spot Instance m5.2xlarge (US East Ohio) on May 8th 2022 : $0.0873 / hour
        price_per_hour = 0.0873
        hour_per_dollar = 1/price_per_hour
        second_per_dollar = hour_per_dollar * 3600

        dict_mean = dict()
        dict_median = dict()
        dict_stddev = dict()
        dict_mean_train_s = dict()
        dict_median_train_s = dict()

        for f in frameworks:
            b = list(results_ranked_df[results_ranked_df['framework'] == f]['time_infer_s'])
            c = list(results_ranked_df[results_ranked_df['framework'] == f]['rows_per_second'])
            # print(f'{f} | {len(b)}')
            b.sort()
            # print(b)
            b_mean = np.mean(b)
            c_mean = np.mean(c)
            c_median = np.median(c)

            dict_mean[f] = b_mean
            b_median = np.median(b)
            dict_median[f] = b_median
            dict_stddev[f] = np.std(b)
            dict_mean_train_s[f] = np.mean(list(results_ranked_df[results_ranked_df['framework'] == f]['time_train_s']))
            dict_median_train_s[f] = np.median(list(results_ranked_df[results_ranked_df['framework'] == f]['time_train_s']))
            rows_per_s_mean = 1/b_mean
            rows_per_s_median = 1/b_median
            print(f'{f}\t| rows_per_s_mean={round(rows_per_s_mean, 2)} | rows_per_s_median={round(rows_per_s_median, 2)}')
            # print(f'true_mean {c_mean}')
            # print(f'true_median {c_median}')
            rows_per_dollar_mean = rows_per_s_mean * second_per_dollar
            rows_per_dollar_median = rows_per_s_median*second_per_dollar
            if rows_per_dollar_median > 1e8:
                print(f'$1 dollar : {round(rows_per_dollar_median / 1e6)}M')
            else:
                print(f'$1 dollar : {round(rows_per_dollar_median/1000)}k')
            # if rows_per_dollar_mean > 1e8:
            #     print(f'$1 dollar mean: {round(rows_per_dollar_mean / 1e6)}M')
            # else:
            #     print(f'$1 dollar mean: {round(rows_per_dollar_mean/1000)}k')

        hue_rename_dict = {
            'Ensemble_AG_FTT_all_bq_mytest24h_2022_09_14_v3': 'AutoGluon Experimental (v0.7), GPU, 24hr',
            'Ensemble_AG_FTT_all_bq_mytest4h_2022_09_14_v2': 'AutoGluon Experimental (v0.7), GPU, 4hr',
            'Ensemble_AG_bq_mytest4h_2022_09_14_v2': 'AutoGluon Best (v0.6), 4hr',
            'AutoGluon_bq_1h8c_2022_06_26_binary': 'AutoGluon Best (v0.6)',
            'AutoGluon_hq_1h8c_2022_06_26_binary': 'AutoGluon High (v0.6)',
            'AGv053_Jul30_high_il0_01_1h8c_2022_07_31_i01': 'AutoGluon High (v0.6), infer_limit=0.01',
            'AGv053_Jul30_high_il0_005_1h8c_2022_07_31_i005': 'AutoGluon High (v0.6), infer_limit=0.005',
            'AGv053_Jul30_high_il0_002_1h8c_2022_07_31_i002': 'AutoGluon High (v0.6), infer_limit=0.002',
            'AutoGluon_benchmark_1h8c_gp3_2022_jmlr': 'AutoGluon Best (v0.3.1)',
            # 'AutoGluon_bestquality_1h_2021_09_02': 'AutoGluon Best (v0.3.1)',
            'AutoGluon_bestquality_1h_2021_02_06_v0_1_0': 'AutoGluon Best (v0.1.0)',
            'AutoGluon_mq_4h64c_2022_06_21_CatBoost': 'CatBoost (AutoGluon v0.6)',
            'AutoGluon_mq_4h64c_2022_06_21_LightGBM': 'LightGBM (AutoGluon v0.6)',
            'AutoGluon_mq_4h64c_2022_06_21_XGBoost': 'XGBoost (AutoGluon v0.6)',
            'GAMA_benchmark_1h8c_gp3_2022_jmlr': 'GAMA',
            'H2OAutoML_1h8c_gp3_2022_jmlr': 'H2OAutoML',
            'TPOT_1h8c_gp3_2022_jmlr': 'TPOT',
            'TunedRandomForest_1h8c_gp3_2022_jmlr': 'TunedRandomForest',
            'autosklearn_1h8c_gp3_2022_jmlr': 'autosklearn',
            'flaml_1h8c_gp3_2022_jmlr': 'flaml',
            'lightautoml_1h8c_gp3_2022_jmlr': 'lightautoml',
            'mljarsupervised_benchmark_1h8c_gp3_2022_jmlr': 'mljarsupervised',
        }
        framework_order = list(hue_rename_dict.values())
        framework_order = [
            "AutoGluon 1.0 (Best, 4h8c)",
            "AutoGluon 1.0 (Best, 1h8c)",
            "AutoGluon 1.0 (Best, 30m8c)",
            "AutoGluon 1.0 (Best, 10m8c)",
            "AutoGluon 1.0 (Best, 5m8c)",
            "AutoGluon 1.0 (High, 4h8c)",
            "AutoGluon 1.0 (High, 4h8c, infer_limit=0.001)",
            "AutoGluon 1.0 (High, 4h8c, infer_limit=0.0005)",
            "AutoGluon 1.0 (High, 4h8c, infer_limit=0.0001)",
            "AutoGluon 0.8.2 (Best, 4h8c)",
            "AutoGluon 0.8.2 (High, 4h8c)",
        ] + framework_order

        results_ranked_overall_df['Framework'] = results_ranked_overall_df['framework'].map(hue_rename_dict).fillna(results_ranked_overall_df['framework'])
        framework_vals = list(results_ranked_overall_df['Framework'].values)
        framework_order = [f for f in framework_order if f in framework_vals]
        framework_other = [f for f in framework_vals if f not in framework_order]
        framework_other = sorted(framework_other)
        framework_order += framework_other
        results_ranked_overall_df['Framework'] = pd.Categorical(results_ranked_overall_df['Framework'], framework_order)
        results_ranked_overall_df = results_ranked_overall_df.sort_values(['Framework'])
        # results_ranked_overall_df = results_ranked_overall_df.sort_values(['Framework'], ascending=True)

        plotter = Plotter(
            results_ranked_fillna_df=results_ranked_fillna_df,
            results_ranked_df=results_ranked_df,
            results_ranked_overall_df=results_ranked_overall_df,
        )

        plotter.plot_boxplot_rescaled_accuracy()
        plotter.plot_boxplot_champion_loss_delta()
        plotter.plot_boxplot_rank()
        plotter.plot_pareto_infer_time(save_prefix=results_dir_output)
        plotter.plot_pareto_train_time(save_prefix=f"{results_dir_output}train_time_")

        # from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
        # task_metadata = load_task_metadata()
        # task_metadata['tid'] = task_metadata['tid'].astype(str)
        # task_metadata['dataset'] = task_metadata['name']
        # plot_1(results_ranked_df, task_metadata=task_metadata)

    from autogluon.common.utils.s3_utils import upload_s3_folder, upload_file

    # upload_s3_folder(bucket="autogluon-zeroshot", prefix="autogluon_v1/", folder_to_upload="data/results/output/openml/autogluon_v1/")
    import shutil

    shutil.make_archive("results", 'zip', "data/results/output/openml/autogluon_v1")
    upload_file(file_name="results.zip", bucket="autogluon-zeroshot", prefix="autogluon_v1")
