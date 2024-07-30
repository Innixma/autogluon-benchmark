from __future__ import annotations

import pandas as pd

from autogluon.common.savers import save_pd
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
from autogluon_benchmark.evaluation.evaluator import Evaluator
from autogluon_benchmark.plotting.plotter import Plotter
from autogluon_benchmark.tasks.experiment_utils import run_experiments


def get_tiny_task_metadata_w_numeric(task_metadata: pd.DataFrame):
    """
    Require at least 1 numeric feature so KNN doesn't crash
    """
    task_metadata_tiny = task_metadata[task_metadata['NumberOfInstances'] <= 20000]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfFeatures'] <= 1000]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfClasses'] <= 100]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfNumericFeatures'] >= 1]
    return task_metadata_tiny


if __name__ == '__main__':
    expname = "./results_sklearnex"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    task_metadata = load_task_metadata('task_metadata.csv')
    task_metadata_tiny = get_tiny_task_metadata_w_numeric(task_metadata)
    tids = task_metadata_tiny["tid"].to_list()

    # tids = tids[:30]  # TODO: This is for demonstration purposes, comment this out to train on more datasets

    # tasks with only boolean numeric features
    banned_tids = [233215]
    tids = [t for t in tids if t not in banned_tids]
    # tids = tids[:15]

    names = [task_metadata[task_metadata["tid"] == t]["name"].iloc[0] for t in tids]

    for i in range(len(tids)):
        print(f"{tids[i]}\t| {names[i]}")

    folds = [0, 1]  # How many folds ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] is all folds), more folds = less noise in results

    model_types = [
        "KNN",
        "RF",
        "XT",
        "LR",
        # "NN_TORCH",
    ]

    # Notes from sklearnex:
    # LR -> Maybe will be faster on larger datasets
    # XT -> Add sklearnex support

    model_types_skex = [
        "KNN",
        "RF",
        "XT",
        "LR",
    ]

    model_types_onnx = [
        "RF",
        "XT",
        "NN_TORCH",
    ]

    methods_dict = {}

    model_types_per = {}

    for model in model_types:
        method_dict = {}
        method_dict[model] = {
            "hyperparameters": {
                model: {"ag.use_daal": False},
            }
        }
        if model in model_types_skex:
            method_dict[f"{model}_EX"] = {
                "hyperparameters": {
                    # TODO: "ag.use_sklearnex": True
                    model: {"ag.use_daal": True},
                }
            }
        if model in model_types_onnx:
            method_dict[f"{model}_ONNX"] = {
                "hyperparameters": {
                    model: {"ag.compile": {"compiler": "onnx"}},
                }
            }
        model_types_per[model] = list(method_dict.keys())
        methods_dict.update(method_dict)


    shared_args = dict(
        time_limit=None,
        fit_weighted_ensemble=False,
        verbosity=1,
    )
    for key in methods_dict:
        methods_dict[key].update(shared_args)

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

    results_df = results_df.rename(columns=dict(
        time_fit="time_train_s",
        time_predict="time_infer_s",
        test_error="metric_error",
        eval_metric="metric",
    ))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)

    evaluator = Evaluator(
        # frameworks=frameworks_run,
        task_metadata=task_metadata,
        # treat_folds_as_datasets=treat_folds_as_datasets,
    )

    evaluator_output = evaluator.transform(data=results_df)

    results_ranked_df = evaluator_output.results_ranked

    output_path = f"{expname}/output"
    figure_savedir = f"{output_path}/figures"
    save_pd.save(path=f"{output_path}/results.csv", df=results_df)
    save_pd.save(path=f"{output_path}/results_ranked_agg.csv", df=evaluator_output.results_ranked_agg)
    save_pd.save(path=f"{output_path}/results_ranked.csv", df=results_ranked_df)

    plotter = Plotter(
        results_ranked_fillna_df=results_ranked_df,
        results_ranked_df=results_ranked_df,
        save_dir=figure_savedir,
        show=False,
    )

    plotter.plot_all(
        # calibration_framework="RandomForest (2023, 4h8c)",
        calibration_elo=1000,
        BOOTSTRAP_ROUNDS=100,  # Reduce this to lower values for a faster execution. Use 1000 for the final plot.
        plot_critical_difference=False,
    )

    results_list = []

    for model_type in model_types:
        frameworks_run = model_types_per[model_type]

        evaluator = Evaluator(
            frameworks=frameworks_run,
            frameworks_compare_vs_all=[model_type],
            task_metadata=task_metadata,
        )

        evaluator_output = evaluator.transform(data=results_df)

        results_ranked_df = evaluator_output.results_ranked

        results_vs_df = evaluator_output.results_pairs_merged_dict[model_type]
        results_vs_df = results_vs_df[results_vs_df["framework"] != model_type]
        results_list.append(results_vs_df)

        plotter = Plotter(
            results_ranked_fillna_df=results_ranked_df,
            results_ranked_df=results_ranked_df,
            save_dir=f"{figure_savedir}/{model_type}",
            show=False,
        )

        plotter.plot_all(
            # calibration_framework="RandomForest (2023, 4h8c)",
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=100,  # Reduce this to lower values for a faster execution. Use 1000 for the final plot.
            plot_critical_difference=False,
            plot_elo_ratings=False,
        )

    a = pd.concat(results_list)

    a = a[[
        "framework",
        "Winrate",
        ">",
        "<",
        "=",
        "% Loss Reduction",
        "% Loss Reduction (median)",
        "Avg Fit Speed Diff",
        "Avg Inf Speed Diff",
    ]]

    # TODO: Add check that sklearnex is installed
    # TODO: Add infer_batch sizes
    # TODO: Add compiled versions
    # TODO: Median fit speed diff / inf speed diff?
    # TODO: fit speed diff / inf speed diff boxplot?
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(a)
