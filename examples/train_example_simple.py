from __future__ import annotations

import pandas as pd

from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
from autogluon_benchmark.tasks.experiment_utils import run_experiments


if __name__ == '__main__':
    expname = "./results_simple"  # folder location of all experiment artifacts
    ignore_cache = True  # set to True to overwrite existing caches and re-run experiments from scratch
    task_metadata = load_task_metadata('task_metadata.csv')

    tids = [2073, 146818]
    folds = [0]  # How many folds ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] is all folds), more folds = less noise in results

    methods_dict = {
        "RF": {
            "hyperparameters": {"RF": {}},
            "fit_weighted_ensemble": False,
        }
    }
    methods = list(methods_dict.keys())

    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        methods_dict=methods_dict,
        task_metadata=task_metadata,
        ignore_cache=ignore_cache,

        # Uncomment if you want to get a results_lst that isn't a list of pandas DataFrames but arbitrary objects.
        #  Note that this will remove the caching functionality, as it falls back to "DummyExperiment" class.
        #  To retain the caching functionality, implement a custom `Experiment` class that can cache the custom object.
        # cache_class=None,
        # cache_class_kwargs=...,

        # Set exec_func if you want to do logic different from `fit_ag`, such as returning a non-DataFrame output artifact.
        # exec_func=...,
        # exec_func_kwargs=...,
    )
    results_df = pd.concat(results_lst, ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)
