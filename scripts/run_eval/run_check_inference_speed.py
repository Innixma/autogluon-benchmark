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


def plot_2(data, infer_speed_multiplier=None):
    results_ranked_overall_df = data.copy()
    x_name = 'Predict Speed Per-Row (seconds) (median)'
    y_name = 'Normalized Result (Score)'
    results_ranked_overall_df[y_name] = 1 - results_ranked_overall_df['loss_rescaled']
    results_ranked_overall_df[x_name] = [dict_median[z[0]] for z in zip(results_ranked_overall_df['framework'])]
    Xs = list(results_ranked_overall_df["Predict Speed Per-Row (seconds) (median)"])
    Ys = list(results_ranked_overall_df["Normalized Result (Score)"])
    # plot_pareto_frontier(Xs=Xs, Ys=Ys, maxX=False, maxY=True)
    pareto_front = get_pareto_frontier(Xs=Xs, Ys=Ys, max_X=False, max_Y=True)

    if infer_speed_multiplier is not None:
        print(f'Altering inference speed by multipler: {infer_speed_multiplier}')
        results_ranked_overall_df[x_name] /= infer_speed_multiplier
    g = sns.relplot(
        x=x_name,
        y=y_name,
        data=results_ranked_overall_df,
        palette='Paired',
        hue="Framework",
        height=10,
        s=300,
        # aspect=1.6,
    )
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y)

    plt.ylim(None, 1)
    g.set(xscale="log")
    fig = g.fig

    # Add a title to the Figure
    fig.suptitle(f"AutoMLBenchmark 2023 Score vs Inference Throughput (104 datasets, 10-fold, 1 hour, 8 cores)", fontsize=16)
    plt.show()


if __name__ == '__main__':
    # results_dir = "s3://autogluon-zeroshot/autogluon_v1/"  # s3 path
    results_dir = 'data/results/output/openml/autogluon_v1/'  # local path
    results_dir_input = results_dir
    results_dir_output = results_dir
    problem_type = 'all'
    run_path_prefix = f'4h8c/{problem_type}/'
    run_path_prefix_overall = f'4h8c_fillna/{problem_type}/'

    results_ranked_df = load_pd.load(f'{results_dir_input}{run_path_prefix}results_ranked_by_dataset_valid.csv')
    results_ranked_overall_df = load_pd.load(f'{results_dir_input}{run_path_prefix_overall}results_ranked_valid.csv')

    results_ranked_df['dataset'] = results_ranked_df['dataset'].astype(str)
    results_ranked_df['rows_per_second'] = 1 / results_ranked_df['time_infer_s']

    a = list(results_ranked_df['framework'].unique())
    print(a)

    # Price Spot Instance m5.2xlarge (US East Ohio) on May 8th 2022 : $0.0873 / hour
    price_per_hour = 0.0873
    hour_per_dollar = 1/price_per_hour
    second_per_dollar = hour_per_dollar * 3600

    dict_mean = dict()
    dict_median = dict()

    for f in a:
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
    results_ranked_overall_df['Framework'] = results_ranked_overall_df['framework'].map(hue_rename_dict).fillna(results_ranked_overall_df['framework'])
    framework_vals = list(results_ranked_overall_df['Framework'].values)
    framework_order = [f for f in framework_order if f in framework_vals]
    framework_other = [f for f in framework_vals if f not in framework_order]
    framework_order += framework_other
    results_ranked_overall_df['Framework'] = pd.Categorical(results_ranked_overall_df['Framework'], framework_order)
    results_ranked_overall_df = results_ranked_overall_df.sort_values(['Framework'])

    # results_ranked_overall_df = results_ranked_overall_df.sort_values(['Framework'], ascending=True)

    plot_2(results_ranked_overall_df, infer_speed_multiplier=None)

    from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
    task_metadata = load_task_metadata()
    task_metadata['tid'] = task_metadata['tid'].astype(str)
    task_metadata['dataset'] = task_metadata['name']
    plot_1(results_ranked_df, task_metadata=task_metadata)
