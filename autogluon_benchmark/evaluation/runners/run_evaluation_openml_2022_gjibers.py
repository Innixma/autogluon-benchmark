"""
TODO: Remove code dupe with `run_evaluation_openml.py`
"""
from autogluon_benchmark.evaluation import evaluate_results
from autogluon_benchmark.evaluation.constants import TIME_INFER_S
from autogluon_benchmark.evaluation.evaluate_utils import compute_stderr_z_stat, compute_stderr_z_stat_bulk, compute_win_rate_per_dataset, graph_vs
from autogluon_benchmark.evaluation import BenchmarkEvaluator


def run(
    *,
    frameworks_run,
    paths,
    output_suffix='ag_full_v5/1h8c',
    framework_nan_fill=None,
    problem_type=None,
    folds_to_keep: list = None,
    compute_z_score=True,
    treat_folds_as_datasets=False,
    banned_datasets=None,
    infer_batch_size=None,
    clean_data=True,
    use_tid_as_dataset_name=True,
    filter_errors=False,  # If True, all dataset errors will be filtered out
):
    results_dir = 'data/results/'
    if folds_to_keep is None:
        folds_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    frameworks_compare_vs_all = []
    if len(frameworks_compare_vs_all) == 0:
        frameworks_compare_vs_all = [frameworks_run[0]]

    benchmark_evaluator = BenchmarkEvaluator(
        results_dir=results_dir,
        output_suffix=output_suffix,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        framework_nan_fill=framework_nan_fill,
        filter_errors=filter_errors,
    )

    results_raw = benchmark_evaluator.load_data(paths=paths,
                                                frameworks=frameworks_run,
                                                folds=folds_to_keep,
                                                clean_data=clean_data,
                                                problem_type=problem_type,
                                                banned_datasets=banned_datasets,
                                                infer_batch_size=infer_batch_size,
                                                treat_folds_as_datasets=treat_folds_as_datasets)

    folds_to_keep = sorted(results_raw['fold'].unique())

    if len(folds_to_keep) > 1:
        compute_win_rate_per_dataset(f1=frameworks_run[0], f2=frameworks_run[1], results_raw=results_raw, folds=folds_to_keep)
    if compute_z_score and len(folds_to_keep) > 1:
        z_stat_df = compute_stderr_z_stat_bulk(framework=frameworks_run[0], frameworks_to_compare=frameworks_run[1:], results_raw=results_raw)
        z_stat_series = compute_stderr_z_stat(results_raw, f1=frameworks_run[0], f2=frameworks_run[1], folds=folds_to_keep, verbose=False)
        graph_vs(results_df=results_raw, f1=frameworks_run[0], f2=frameworks_run[1], z_stats=z_stat_series)

    results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
        results_raw=results_raw,
        frameworks=frameworks_run,
        columns_to_agg_extra=[
            TIME_INFER_S,
        ],
        frameworks_compare_vs_all=frameworks_compare_vs_all,
        output_dir=benchmark_evaluator.results_dir_output,
    )


if __name__ == '__main__':

    framework_name_suffix = '_4h8c_gp3_2022_jmlr'
    frameworks_run = [
        'AutoGluon_benchmark',
        'autosklearn',
        # 'autosklearn2',
        'flaml',
        'GAMA_benchmark',
        'H2OAutoML',
        'lightautoml',
        'mljarsupervised_benchmark',
        'TPOT',
        'TunedRandomForest',
        'RandomForest',
        # 'mlr3automl',
    ]
    frameworks_run = [f + framework_name_suffix for f in frameworks_run]

    paths = [
        'amlb/2022_jmlr.csv',  # gjibers et al
    ]

    use_tid_as_dataset_name = False
    problem_types = ['binary', 'multiclass', 'regression']
    treat_folds_as_datasets = False
    # filter_errors = True
    infer_batch_size = None
    run(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=f'2022_amlb_jmlr/1h8c/all',
        problem_type=problem_types,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=True,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
    )
    run(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=f'2022_amlb_jmlr/1h8c_fillna/all',
        framework_nan_fill='constantpredictor_1h8c_gp3_2022_jmlr',
        problem_type=problem_types,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=False,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        compute_z_score=False,
    )
