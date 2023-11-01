import pandas as pd

from autogluon.common.savers import save_pd
from autogluon_benchmark import AutoGluonTaskWrapper
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
from autogluon_benchmark.evaluation.evaluate_utils import compare_frameworks


metrics_dict = {
    "binary": "roc_auc",
    "multiclass": "log_loss",
    "regression": "rmse",
}


if __name__ == '__main__':
    task_metadata = load_task_metadata('task_metadata.csv')
    task_metadata_tiny = task_metadata[task_metadata['NumberOfInstances'] <= 2000]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfFeatures'] <= 100]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfClasses'] <= 10]
    tids = task_metadata_tiny["tid"].to_list()
    print(tids)

    tids = tids[:2]  # TODO: This is for demonstration purposes, comment this out to train on more datasets
    folds = [0]  # How many folds ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] is all folds), more folds = less noise in results

    frameworks_dict = dict(
        FTT={
            "hyperparameters": {
                "FT_TRANSFORMER": {  # # FT_TRANSFORMER hyperparameters go here

                },
            },
        },
        DUMMY={  # Constant Predictor
            "hyperparameters": {
                "DUMMY": {},
            },
        },
        GBM={  # LightGBM
            "hyperparameters": {
                "GBM": {},
            },
        },
        FASTAI={  # FastAI Tabular Neural Network
            "hyperparameters": {
                "FASTAI": {},
            },
        },
    )
    shared_args = dict(
        time_limit=3600,
        fit_weighted_ensemble=False,
    )
    for key in frameworks_dict:
        frameworks_dict[key].update(shared_args)

    frameworks = ["DUMMY", "FTT", "GBM", "FASTAI"]

    result_lst = []

    for tid in tids:
        task = AutoGluonTaskWrapper.from_task_id(task_id=tid)
        task_name = task_metadata[task_metadata["tid"] == tid]["name"].iloc[0]
        for fold in folds:
            for framework in frameworks:
                fit_args = frameworks_dict[framework]
                print(f"Running Task Name: '{task_name}'")
                out = task.run(fold=fold, fit_args=fit_args)

                out["framework"] = framework
                out["dataset"] = task_name
                print(f"Task  Name: {out['dataset']}")
                print(f"Task    ID: {out['tid']}")
                print(f"Metric    : {out['eval_metric']}")
                print(f"Test Score: {out['test_score']:.4f}")
                print(f"Val  Score: {out['val_score']:.4f}")
                print(f"Test Error: {out['test_error']:.4f}")
                print(f"Val  Error: {out['val_error']:.4f}")
                print(f"Fit   Time: {out['time_fit']:.3f}s")
                print(f"Infer Time: {out['time_predict']:.3f}s")

                out.pop("predictions")
                out.pop("probabilities")
                out.pop("truth")
                out.pop("others")

                result_lst.append(out)

    df_results = pd.DataFrame(result_lst)
    ordered_columns = ["dataset", "fold", "framework", "test_error", "val_error", "eval_metric", "test_score", "val_score", "time_fit"]
    columns_reorder = ordered_columns + [c for c in df_results.columns if c not in ordered_columns]
    df_results = df_results[columns_reorder]
    df_results = df_results.rename(columns=dict(
        time_fit="time_train_s",
        time_predict="time_infer_s",
        test_error="metric_error",
    ))

    save_pd.save(path=f"./results.csv", df=df_results)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_results)

    results_ranked, results_ranked_by_dataset = compare_frameworks(
        results_raw=df_results,
        columns_to_agg_extra=["time_infer_s"],
    )
    save_pd.save(path=f"./results_ranked.csv", df=results_ranked)
    save_pd.save(path=f"./results_ranked_by_dataset.csv", df=results_ranked_by_dataset)
