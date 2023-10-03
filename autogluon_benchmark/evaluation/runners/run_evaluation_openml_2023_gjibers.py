import pandas as pd

from autogluon.bench.eval.scripts.run_evaluation_openml import evaluate
from autogluon.bench.eval.evaluation.benchmark_evaluator import BenchmarkEvaluator


if __name__ == '__main__':
    framework_name_suffix = '_1h8c_gp3_amlb_2023'
    frameworks_run = [
        'AutoGluon_benchmark',
        'AutoGluon_hq',
        'autosklearn',
        'flaml',
        'H2OAutoML',
        'lightautoml',
        'GAMA_benchmark',
        # 'mljarsupervised_benchmark',
        # 'mljarsupervised_perform',
        # 'TPOT',
        # 'TunedRandomForest',
        # 'RandomForest',
        # 'NaiveAutoML',
    ]
    frameworks_run = [f + framework_name_suffix for f in frameworks_run]

    paths = [
        "s3://automl-benchmark-ag/aggregated/amlb/amlb_2023_preprocessed.csv",
    ]

    problem_types = ['binary', 'multiclass', 'regression']
    use_tid_as_dataset_name = False
    treat_folds_as_datasets = False
    task_metadata = "task_metadata.csv"
    infer_batch_size = None
    evaluate(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=f'2023_amlb_jmlr/1h8c/all',
        problem_type=problem_types,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=False,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        task_metadata=task_metadata,
    )
    evaluate(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=f'2023_amlb_jmlr/1h8c_fillna/all',
        framework_nan_fill='constantpredictor_1h8c_gp3_amlb_2023',
        problem_type=problem_types,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=False,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        compute_z_score=False,
        task_metadata=task_metadata,
    )
