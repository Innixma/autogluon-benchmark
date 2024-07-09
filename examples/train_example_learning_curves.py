from __future__ import annotations

import pandas as pd

from autogluon.common.savers import save_pd
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
from autogluon_benchmark.evaluation.evaluate_utils import compare_frameworks
from autogluon_benchmark.tasks.experiment_utils import run_experiments


def get_tiny_task_metadata(task_metadata: pd.DataFrame):
    task_metadata_tiny = task_metadata[task_metadata['NumberOfInstances'] <= 2000]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfFeatures'] <= 100]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfClasses'] <= 10]
    return task_metadata_tiny


if __name__ == '__main__':
    expname = "./results_learning_curves"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    task_metadata = load_task_metadata('task_metadata.csv')
    task_metadata_tiny = get_tiny_task_metadata(task_metadata)
    tids = task_metadata_tiny["tid"].to_list()

    tids = tids[:2]  # TODO: This is for demonstration purposes, comment this out to train on more datasets
    folds = [0, 1, 2]  # How many folds ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] is all folds), more folds = less noise in results
    methods = ["EXAMPLE"]

    methods_dict = dict(
        EXAMPLE={
            "hyperparameters": {
                "GBM": {
                    "ag.early_stop": 999999,
                    "num_boost_round": 1000,
                },
                "XGB": {
                    "ag.early_stop": 999999,
                    "n_estimators": 1000,
                },
                # "CAT": {},
                "NN_TORCH": {
                    "epochs_wo_improve": 999999,
                    "num_epochs": 100,
                },
                # "FASTAI": {},
            },
            # "verbosity": 4,
            # "learning_curves": True,
            # (you will need a code edit to ensure the test_data is passed in)
        },
    )
    shared_args = dict(
        time_limit=None,
        fit_weighted_ensemble=False,
    )
    for key in methods_dict:
        methods_dict[key].update(shared_args)

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
    df_results = pd.concat(results_lst, ignore_index=True)

    df_results = df_results.rename(columns=dict(
        time_fit="time_train_s",
        time_predict="time_infer_s",
        test_error="metric_error",
    ))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_results)

    results_ranked, results_ranked_by_dataset = compare_frameworks(
        results_raw=df_results,
        columns_to_agg_extra=["time_infer_s"],
    )

    output_path = f"{expname}/output"
    save_pd.save(path=f"{output_path}/results.csv", df=df_results)
    save_pd.save(path=f"{output_path}/results_ranked.csv", df=results_ranked)
    save_pd.save(path=f"{output_path}/results_ranked_by_dataset.csv", df=results_ranked_by_dataset)
