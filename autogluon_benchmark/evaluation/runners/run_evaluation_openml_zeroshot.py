from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd

from autogluon.common.savers import save_pd

from autogluon_benchmark.evaluation import evaluate_results
from autogluon_benchmark.evaluation.constants import TIME_INFER_S
from autogluon_benchmark.evaluation import BenchmarkEvaluator


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

    results_raw = benchmark_evaluator.load_data(
        paths=paths,
        frameworks=frameworks_run,
        folds=folds_to_keep,
        clean_data=clean_data,
        problem_type=problem_type,
        banned_datasets=banned_datasets,
        infer_batch_size=None,
        treat_folds_as_datasets=treat_folds_as_datasets,
    )

    # results_raw = results_raw[results_raw['tid'].isin(valid_tids)]

    if frameworks_rename_dict:
        results_raw['framework'] = results_raw['framework'].map(frameworks_rename_dict).fillna(results_raw['framework'])
        frameworks_run = [frameworks_rename_dict.get(f, f) for f in frameworks_run]

    folds_to_keep = sorted(results_raw['fold'].unique())
    from autogluon_benchmark.evaluation.evaluate_utils import compute_stderr_z_stat, compute_stderr_z_stat_bulk, \
        compute_win_rate_per_dataset, graph_vs
    if len(folds_to_keep) > 1:
        compute_win_rate_per_dataset(f1=frameworks_run[0], f2=frameworks_run[1], results_raw=results_raw, folds=folds_to_keep)
    if compute_z_score and len(folds_to_keep) > 1:
        z_stat_df = compute_stderr_z_stat_bulk(framework=frameworks_run[0], frameworks_to_compare=frameworks_run[1:], results_raw=results_raw)
        z_stat_series = compute_stderr_z_stat(results_raw, f1=frameworks_run[0], f2=frameworks_run[1], folds=folds_to_keep, verbose=False)
        graph_vs(results_df=results_raw, f1=frameworks_run[0], f2=frameworks_run[1], z_stats=z_stat_series)

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


if __name__ == '__main__':
    ts = time.time()
    problem_types = ['binary', 'multiclass', 'regression']

    treat_folds_as_datasets = True
    folds_to_keep = [0]

    s3_input_dir = 's3://automl-benchmark-ag/aggregated/ec2'
    paths = [
        # f'{s3_input_dir}/2023_07_25/leaderboard_preprocessed.csv',
        f'{s3_input_dir}/2023_07_25/results_preprocessed.csv',
    ]

    paths2 = [
        f'{s3_input_dir}/2023_07_25/leaderboard_preprocessed.csv',
    ]
    results_raw_baselines = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=True)
    results_raw_configs = pd.concat([pd.read_csv(path) for path in paths2], ignore_index=True, sort=True)

    # results_raw = results_raw[results_raw['framework'].str.contains('_60h8c_')]
    # results_raw = results_raw[results_raw['framework'].str.contains('AutoGluon')]
    results_raw_baselines = results_raw_baselines[~results_raw_baselines['framework'].str.contains('ZS_BAG_')]
    # results_raw = results_raw[results_raw['framework'].str.contains('_1h8c_')]
    results_raw_configs = results_raw_configs[results_raw_configs['framework'].str.contains('ZS_BAG_')]
    results_raw_configs = results_raw_configs.drop_duplicates(subset=['framework', 'dataset', 'fold'])

    path_leaderboard_preprocessed_configs = f'{s3_input_dir}/2023_07_25/leaderboard_preprocessed_configs.csv'
    save_pd.save(path=path_leaderboard_preprocessed_configs, df=results_raw_configs)

    raise AssertionError()

    # results_raw2 = results_raw2[results_raw2['framework'].str.contains('_c1')]

    # paths = [
    #     f'{s3_input_dir}/2023_07_25/leaderboard_preprocessed.csv',
    #     f'{s3_input_dir}/2023_07_25/results_preprocessed.csv',
    # ]

    banned_frameworks = [
        "constantpredictor_1h8c_2023_07_25",
        "constantpredictor_4h8c_2023_07_25",
        "NaiveAutoML_1h8c_2023_07_25",
        "NaiveAutoML_4h8c_2023_07_25",
        "mlr3automl_4h8c_2023_07_25",
        "mlr3automl_1h8c_2023_07_25",
        "AutoWEKA_4h8c_2023_07_25",
        "AutoWEKA_1h8c_2023_07_25",
        "TPOT_4h8c_2023_07_25",
        "TPOT_1h8c_2023_07_25",
    ]

    frameworks_run_baselines = list(results_raw_baselines['framework'].unique())
    frameworks_run_baselines = [f for f in frameworks_run_baselines if f not in banned_frameworks]

    run(
        frameworks_run=frameworks_run_baselines,
        paths=results_raw_baselines,
        output_path='s3://automl-benchmark-ag/aggregated/ec2/2023_07_25/evaluation/compare/',
        folds_to_keep=folds_to_keep,
        treat_folds_as_datasets=treat_folds_as_datasets,
    )

    results_raw = pd.concat([results_raw_baselines, results_raw_configs], ignore_index=True)

    a = results_raw['dataset'].value_counts()

    max_failures = 16

    dense_datasets = None
    for i in range(0):
        dataset_counts = results_raw['dataset'].value_counts()
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
            frameworks_to_keep = results_raw[results_raw['dataset'] == dataset_to_fix]['framework'].unique()
            frameworks_to_keep_set = set(frameworks_to_keep)
            frameworks_dropped = [f for f in frameworks if f not in frameworks_to_keep]
            print(f'Dropping frameworks: {frameworks_dropped}')
            results_raw = results_raw[results_raw['framework'].isin(frameworks_to_keep)]

    if dense_datasets is not None:
        results_raw = results_raw[results_raw['dataset'].isin(dense_datasets)]

    path_leaderboard_pruned = f'{s3_input_dir}/2023_07_25/leaderboard_pruned.csv'
    from autogluon.common.savers import save_pd
    save_pd.save(path=path_leaderboard_pruned, df=results_raw)

    frameworks_run = list(results_raw['framework'].unique())
    frameworks_run_zeroshot = [zs_name for zs_name in frameworks_run if 'ZS_BAG_' in zs_name]
    # frameworks_run_zeroshot = frameworks_run_zeroshot[:400]

    # FIXME: TEMP
    constraint = '60h8c'
    constraint_str = f'_{constraint}'
    run_name = '2023_07_25'
    run_name_str = f'_{run_name}'
    # FIXME: TEMP

    frameworks_run_zeroshot_rename = {f: f.rsplit(f'{constraint_str}{run_name_str}_', 1)[-1] for f in
                                      frameworks_run_zeroshot}

    # banned_frameworks += frameworks_run_zeroshot

    frameworks_compare_vs_all = [
        "AutoGluon_bq_4h8c_2023_07_25",
        "AutoGluon_bq_1h8c_2023_07_25",
        "AutoGluon_ezsh_4h8c_2023_07_25",
        "AutoGluon_ezsh_1h8c_2023_07_25",
    ]

    run(
        paths=results_raw_baselines,
        frameworks_run=frameworks_run_zeroshot,
        frameworks_rename_dict=frameworks_run_zeroshot_rename,
        # frameworks_compare_vs_all=frameworks_compare_vs_all,
        output_path='s3://automl-benchmark-ag/aggregated/ec2/2023_07_25/evaluation/configs/',
        folds_to_keep=folds_to_keep,
        treat_folds_as_datasets=treat_folds_as_datasets,
    )

    te = time.time()
    print(f'Time Taken to Evaluate: {te-ts:.2f}s')
