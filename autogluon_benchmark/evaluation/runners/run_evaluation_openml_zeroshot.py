from __future__ import annotations

import copy
import time
from typing import Dict, List

import pandas as pd

from autogluon.common.savers import save_pd

from autogluon.bench.eval.evaluation import evaluate_results
from autogluon.bench.eval.evaluation.constants import TIME_INFER_S
from autogluon.bench.eval.evaluation.benchmark_evaluator import BenchmarkEvaluator

from autogluon_benchmark.evaluation.preprocess.clean_results_util import clean_data


def run(
    *,
    frameworks_run: List[str],
    paths: List[str] | pd.DataFrame,
    output_suffix: str = 'ag_full_v5/1h8c',
    output_path: str | None = None,  # TODO: Add to Bench
    frameworks_compare_vs_all: List[str] | None = None,
    framework_nan_fill: str | None = None,
    problem_type: List[str] | str | None = None,
    folds_to_keep: List[int] | None = None,
    compute_z_score: bool = True,
    treat_folds_as_datasets: bool = False,
    banned_datasets: List[str] | None = None,
    infer_batch_size: int | None = None,
    clean_data: bool = True,
    use_tid_as_dataset_name: bool = True,
    filter_errors: bool = False,
    frameworks_rename_dict: Dict[str, str] | None = None,  # TODO: Add to Bench
):
    results_dir = 'data/results/'

    if frameworks_compare_vs_all is None:
        frameworks_compare_vs_all = []

    benchmark_evaluator = BenchmarkEvaluator(
        results_dir=results_dir,
        output_suffix=output_suffix,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        filter_errors=filter_errors,
        task_metadata='task_metadata_244.csv',
    )

    # datasets = benchmark_evaluator.filter_datasets(max_rows=1000)

    results_raw = benchmark_evaluator.load_data(
        paths=paths,
        frameworks=frameworks_run,
        folds=folds_to_keep,
        clean_data=clean_data,
        problem_type=problem_type,
        banned_datasets=banned_datasets,
        infer_batch_size=None,
        treat_folds_as_datasets=treat_folds_as_datasets,
        # valid_datasets=datasets,
    )

    if frameworks_rename_dict:
        results_raw['framework'] = results_raw['framework'].map(frameworks_rename_dict).fillna(results_raw['framework'])
        frameworks_run = [frameworks_rename_dict.get(f, f) for f in frameworks_run]

    if output_path is not None:
        results_dir = output_path

    results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
        results_raw=results_raw,
        frameworks=frameworks_run,
        columns_to_agg_extra=[
            TIME_INFER_S,
        ],
        frameworks_compare_vs_all=frameworks_compare_vs_all,
        output_dir=results_dir,
    )


def filter_partial_failures(results_df: pd.DataFrame, leaderboard_df: pd.DataFrame, dataset_column_name="task") -> pd.DataFrame:
    """
    Filter the results_df by removing tasks which did not have all models in the framework_parent exist in leaderboard_df.

    For example, if a given framework_parent trains 16 models, but only 14 entries exist in leaderboard_df for a given task,
    then filter the result of framework_parent on that task in results_df.

    Parameters
    ----------
    results_df
    leaderboard_df

    Returns
    -------

    """
    results_df_og = results_df
    leaderboard_df = copy.deepcopy(leaderboard_df)
    results_df = copy.deepcopy(results_df)

    leaderboard_df['task_name'] = leaderboard_df['dataset'] + '_' + leaderboard_df['fold'].astype(str)
    results_df['task_name'] = results_df[dataset_column_name] + '_' + results_df['fold'].astype(str)
    results_df['framework_full'] = results_df['framework'] + '_' + results_df['constraint']

    unique_tasks = set(leaderboard_df['task_name'].unique())
    leaderboard_df_dedupe = leaderboard_df[['framework', 'framework_parent']].drop_duplicates()
    parent_to_config_dict = leaderboard_df_dedupe.groupby('framework_parent')['framework'].apply(list).to_dict()
    config_to_task_dict = leaderboard_df[['framework', 'task_name']].groupby('framework')['task_name'].apply(set).to_dict()
    unique_framework_parent = list(parent_to_config_dict.keys())
    unique_configs = list(config_to_task_dict.keys())

    task_failures_dict = dict()
    for config in unique_configs:
        config_tasks = config_to_task_dict[config]
        task_failures = list(unique_tasks.difference(config_tasks))
        task_failures_dict[config] = task_failures

    task_failures_parent_dict = dict()
    c_indices = []
    for framework_parent in unique_framework_parent:
        task_failures_parent_dict[framework_parent] = set()
        configs_in_parent = parent_to_config_dict[framework_parent]
        for config in configs_in_parent:
            task_failures_parent_dict[framework_parent] = task_failures_parent_dict[framework_parent].union(task_failures_dict[config])

        b = leaderboard_df[(leaderboard_df['framework_parent'] == framework_parent) & (leaderboard_df['task_name'].isin(task_failures_parent_dict[framework_parent]))]
        c = results_df[(results_df['framework_full'] == framework_parent) & (results_df['task_name'].isin(task_failures_parent_dict[framework_parent]))]

        c_indices += list(c.index)

        print(f'{len(b)}\t{len(c)}\t{framework_parent}')

    c_indices = set(c_indices)
    results_df_filtered = copy.deepcopy(results_df_og.loc[~results_df_og.index.isin(c_indices)])

    return results_df_filtered


if __name__ == '__main__':
    s3_input_dir = 's3://automl-benchmark-ag/aggregated/ec2'

    path_input_suffix = "2023_08_21"
    path_input = f"{s3_input_dir}/{path_input_suffix}"

    path_output_suffix = "2023_08_21_dummy"
    path_output = f"{s3_input_dir}/{path_output_suffix}"

    ts = time.time()
    problem_types = ['binary', 'multiclass', 'regression']

    treat_folds_as_datasets = True
    folds_to_keep = [0, 1, 2]
    use_tid_as_dataset_name = False

    results_dir = 'data/results/'

    benchmark_evaluator = BenchmarkEvaluator(results_dir=results_dir)

    path_input_suffix = "2023_11_14"
    path_input = f"{s3_input_dir}/{path_input_suffix}"

    paths_baselines = [
        f'{path_input}/results_preprocessed.csv',
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_11_13/results_preprocessed.csv",
    ]

    path_input_suffix = "2023_11_14"
    path_input = f"{s3_input_dir}/{path_input_suffix}"

    paths_configs = [
        f'{path_input}/leaderboard_preprocessed.csv',
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_11_10/leaderboard_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_11_14/leaderboard_preprocessed.csv",
    ]

    results_raw_baselines = benchmark_evaluator.load_results_raw(paths=paths_baselines)
    results_raw_baselines = results_raw_baselines.drop_duplicates(subset=['framework', 'dataset', 'fold'])
    results_csv_configs = results_raw_baselines[results_raw_baselines['framework'].str.contains('ZS_BAG_')]
    results_raw_baselines = results_raw_baselines[~results_raw_baselines['framework'].str.contains('ZS_BAG_')]
    # results_raw_baselines = results_raw_baselines[results_raw_baselines['framework'].str.contains('_1h8c_')]

    run_suffix = "_2023_11_14"

    banned_frameworks = [
        'AutoGluon_ezsh_1h8c',
        'AutoGluon_ezsh_4h8c',
        "constantpredictor_1h8c",
        "constantpredictor_4h8c",
        # 'AutoGluon_bq_24h8c',
        # 'AutoGluon_bq_30m8c',
        # 'AutoGluon_bq_10m8c',
        # 'AutoGluon_bq_5m8c',
        'AutoGluon_ezs_1h8c',
        'AutoGluon_ezs_4h8c',
        'AutoGluon_bq_simple_1h8c',
        'AutoGluon_bq_simple_4h8c',
        # 'AutoGluon_mq_1h8c',
        # 'AutoGluon_mq_4h8c',
        "NaiveAutoML_1h8c",
        "NaiveAutoML_4h8c",
        "mlr3automl_4h8c",
        "mlr3automl_1h8c",
        "AutoWEKA_4h8c",
        "AutoWEKA_1h8c",
        "TPOT_4h8c",
        "TPOT_1h8c",
        # "RandomForest_1h8c",
        # "RandomForest_4h8c",
        # "TunedRandomForest_1h8c",
        # "TunedRandomForest_4h8c",
    ]
    banned_frameworks = [b + run_suffix for b in banned_frameworks]

    frameworks_run_baselines = list(results_raw_baselines['framework'].unique())
    frameworks_run_baselines = [f for f in frameworks_run_baselines if f not in banned_frameworks]

    results_raw_baselines2 = results_raw_baselines[results_raw_baselines["framework"].isin(frameworks_run_baselines)]
    from autogluon.common.savers import save_pd
    save_pd.save(path=f"{path_output}/baselines_raw.csv", df=results_raw_baselines2)

    task_metadata_path = 'task_metadata_244.csv'

    results_raw_baselines_to_save = results_raw_baselines2.copy(deep=True)
    from tabrepo.loaders._results import preprocess_baselines
    results_raw_baselines_to_save = preprocess_baselines(results_raw_baselines_to_save)
    results_raw_baselines_to_save = clean_data(
        results_raw_baselines_to_save,
        task_metadata_path=task_metadata_path,
        convert_infer_time_to_per_row=True,
    )
    results_raw_baselines_to_save = results_raw_baselines_to_save[[
        "dataset",
        "tid",
        "fold",
        "framework",
        "metric_error",
        "metric",
        "problem_type",
        "time_train_s",
        "time_infer_s",
    ]]
    results_raw_baselines_to_save = results_raw_baselines_to_save.sort_values(by=["dataset", "fold", "framework"]).reset_index(drop=True)
    path_baselines_preprocessed = f'{path_output}/baselines.csv'
    save_pd.save(path=path_baselines_preprocessed, df=results_raw_baselines_to_save)
    save_pd.save(path=f'{path_output}/baselines.parquet', df=results_raw_baselines_to_save)


    # run(
    #     frameworks_run=frameworks_run_baselines,
    #     paths=results_raw_baselines2,
    #     output_path=f'{path_output}/evaluation/compare/',
    #     folds_to_keep=folds_to_keep,
    #     treat_folds_as_datasets=treat_folds_as_datasets,
    #     use_tid_as_dataset_name=use_tid_as_dataset_name,
    # )

    results_raw_configs = benchmark_evaluator.load_results_raw(paths=paths_configs)
    results_raw_configs = results_raw_configs[results_raw_configs['framework'].str.contains('ZS_BAG_')]
    results_raw_configs = results_raw_configs[~results_raw_configs['framework'].str.contains('_autogluon_single')]
    results_raw_configs = results_raw_configs.drop_duplicates(subset=['framework', 'dataset', 'fold'])

    print('yo')

    # path_input_suffix = "2023_11_14"
    # path_input = f"{s3_input_dir}/{path_input_suffix}"
    # path_output_suffix = "2023_11_14"
    # path_output = f"{s3_input_dir}/{path_output_suffix}"

    path_results_valid = f'{path_input}/results_valid.csv'
    from autogluon.common.loaders import load_pd
    from autogluon.common.savers import save_pd

    # results_valid_df = load_pd.load(path_results_valid)
    #
    # results_dense_df = filter_partial_failures(
    #     results_df=results_valid_df,
    #     leaderboard_df=results_raw_configs,
    #     dataset_column_name="task",
    # )
    # save_pd.save(path=f'{path_output}/results_valid_dense.csv', df=results_dense_df)

    results_raw_baselines = results_raw_baselines[results_raw_baselines['fold'].isin(folds_to_keep)]

    results_raw_configs = results_raw_configs[results_raw_configs['fold'].isin(folds_to_keep)]

    task_metadata_path = 'task_metadata_244.csv'

    results_raw_configs_to_save = results_raw_configs.copy(deep=True)
    from tabrepo.loaders._results import preprocess_configs
    results_raw_configs_to_save = preprocess_configs(results_raw_configs_to_save)
    results_raw_configs_to_save = clean_data(
        results_raw_configs_to_save,
        task_metadata_path=task_metadata_path,
        convert_infer_time_to_per_row=True,
    )
    results_raw_configs_to_save = results_raw_configs_to_save[[
        "dataset",
        "tid",
        "fold",
        "framework",
        "metric_error",
        "metric_error_val",
        "metric",
        "problem_type",
        "time_train_s",
        "time_infer_s",
    ]]
    results_raw_configs_to_save = results_raw_configs_to_save.sort_values(by=["dataset", "fold", "framework"]).reset_index(drop=True)

    path_leaderboard_preprocessed_configs = f'{path_output}/leaderboard_preprocessed_configs.csv'
    save_pd.save(path=path_leaderboard_preprocessed_configs, df=results_raw_configs_to_save)
    save_pd.save(path=f'{path_output}/configs.parquet', df=results_raw_configs_to_save)



    results_raw = pd.concat([
        # results_raw_baselines,
        results_raw_configs
    ], ignore_index=True)

    # results_raw = results_raw_baselines

    a = results_raw['dataset'].value_counts()

    max_failures = 16

    results_raw['task_name'] = results_raw['dataset'] + '_' + results_raw['fold'].astype(str)

    dense_datasets = None
    for i in range(0):
        dataset_counts = results_raw['task_name'].value_counts()
        total_frameworks = results_raw['framework'].nunique()
        frameworks = list(results_raw['framework'].unique())

        num_failures = total_frameworks - dataset_counts
        dense_datasets = list(num_failures[num_failures == 0].index)
        print(f'num_dense={len(dense_datasets)}')
        print(f'num_frame={total_frameworks}')
        num_failures = num_failures[num_failures != 0]
        num_failures = num_failures.sort_values()

        if len(num_failures) > 0:
            num_failures_to_fix = num_failures.iloc[0]
            if num_failures_to_fix > max_failures:
                print('BREAK')
                break
            dataset_to_fix = num_failures.index[0]
            print(f'Fixing {dataset_to_fix}')
            frameworks_to_keep = results_raw[results_raw['task_name'] == dataset_to_fix]['framework'].unique()
            frameworks_to_keep_set = set(frameworks_to_keep)
            frameworks_dropped = [f for f in frameworks if f not in frameworks_to_keep]
            print(f'Dropping frameworks: {frameworks_dropped}')
            results_raw = results_raw[results_raw['framework'].isin(frameworks_to_keep)]

    if dense_datasets is not None:
        results_raw = results_raw[results_raw['task_name'].isin(dense_datasets)]

    path_leaderboard_pruned = f'{path_output}/leaderboard_pruned.csv'
    from autogluon.common.savers import save_pd
    save_pd.save(path=path_leaderboard_pruned, df=results_raw)

    # frameworks_run = list(results_raw['framework'].unique())
    frameworks_run_zeroshot_all = list(results_raw_configs['framework'].unique())
    frameworks_run_zeroshot = [zs_name for zs_name in frameworks_run_zeroshot_all if 'ZS_BAG_' in zs_name]
    frameworks_run_zeroshot = [zs_name for zs_name in frameworks_run_zeroshot if 'autogluon_single' not in zs_name]

    # FIXME: TEMP
    constraint = '60h8c'
    constraint_str = f'_{constraint}'
    run_name = '2023_08_21'
    run_name_str = f'_{run_name}'

    frameworks_run_zeroshot_rename = {f: f.rsplit(f'{constraint_str}{run_name_str}_', 1)[-1] for f in
                                      frameworks_run_zeroshot}
    # FIXME: TEMP
    constraint = '60h8c_gpu'
    constraint_str = f'_{constraint}'
    run_name = '2023_11_10'
    run_name_str = f'_{run_name}'

    frameworks_run_zeroshot_rename.update({f: f.rsplit(f'{constraint_str}{run_name_str}_', 1)[-1] for f in
                                      frameworks_run_zeroshot})

    # FIXME: TEMP
    constraint = '60h8c'
    constraint_str = f'_{constraint}'
    run_name = '2023_11_14'
    run_name_str = f'_{run_name}'

    frameworks_run_zeroshot_rename.update({f: f.rsplit(f'{constraint_str}{run_name_str}_', 1)[-1] for f in
                                      frameworks_run_zeroshot})

    # banned_frameworks += frameworks_run_zeroshot

    frameworks_compare_vs_all = [
        "AutoGluon_bq_4h8c_2023_07_25",
        "AutoGluon_bq_1h8c_2023_07_25",
        "AutoGluon_ezsh_4h8c_2023_07_25",
        "AutoGluon_ezsh_1h8c_2023_07_25",
    ]

    run(
        paths=results_raw_configs,
        frameworks_run=frameworks_run_zeroshot,
        frameworks_rename_dict=frameworks_run_zeroshot_rename,
        # frameworks_compare_vs_all=frameworks_compare_vs_all,
        output_path=f'{path_output}/evaluation/configs/',
        folds_to_keep=folds_to_keep,
        treat_folds_as_datasets=treat_folds_as_datasets,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
    )

    te = time.time()
    print(f'Time Taken to Evaluate: {te-ts:.2f}s')
