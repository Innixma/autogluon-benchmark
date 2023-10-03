from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

from autogluon.bench.eval.scripts.run_evaluation_openml import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, help="Results Paths", required=True, nargs='+')
    parser.add_argument('--frameworks_run', type=str, help="Name of framework runs", default=None, nargs='+')
    parser.add_argument('--problem_types', type=str, help="Problem types to evaluate", choices=['binary', 'multiclass', 'regression'], default=['binary', 'multiclass', 'regression'], nargs="+")
    parser.add_argument('--folds_to_keep', type=int, help="Folds to keep for evaluation. If None, uses all folds present in the input.", nargs="*")
    parser.add_argument('--filter_errors', type=bool, help="Filter errors during evaluation", default=False)
    parser.add_argument('--treat_folds_as_datasets', type=bool, help="Whether to treat each fold as its own dataset", default=False)
    parser.add_argument('--banned_datasets', type=str, help="Datasets to skip", default=None, nargs='+')
    parser.add_argument('--output_suffix', type=str, help="Directory to save outputs", default="autogluon-bench-dummy")
    parser.add_argument('--task_metadata', type=str, help="The task metadata to filter tasks from, defaults to the AMLB 104 datasets", default="task_metadata.csv")

    args = parser.parse_args()

    frameworks_run = args.frameworks_run
    folds_to_keep = args.folds_to_keep
    problem_type = args.problem_types
    filter_errors = args.filter_errors
    treat_folds_as_datasets = args.treat_folds_as_datasets
    banned_datasets = args.banned_datasets
    output_suffix = args.output_suffix

    # Examples:
    # paths = [
    #   "s3://automl-benchmark-ag/aggregated/ec2/2023_09_25_infoleak/results_preprocessed.csv",
    # ]

    evaluate(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=output_suffix,
        # framework_nan_fill='constantpredictor',
        problem_type=problem_type,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=None,
        filter_errors=filter_errors,
        use_tid_as_dataset_name=False,
        banned_datasets=banned_datasets,
        folds_to_keep=folds_to_keep,
        compute_z_score=True,
        task_metadata='task_metadata.csv',
    )
