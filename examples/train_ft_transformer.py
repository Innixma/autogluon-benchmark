from autogluon_benchmark import AutoGluonTaskWrapper
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata


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

    tid = tids[0]  # Pick a task

    fold = 0

    fit_args = dict(
        hyperparameters={
            'FT_TRANSFORMER': {  # FT_TRANSFORMER hyperparameters go here

            }
        },
        time_limit=3600,
        fit_weighted_ensemble=False,
    )

    for tid in tids:
        task = AutoGluonTaskWrapper.from_task_id(task_id=tid)
        task_name = task_metadata[task_metadata["tid"] == tid]["name"].iloc[0]
        print(f"Running Task Name: '{task_name}'")
        out = task.run(fold=fold, fit_args=fit_args)

        print(f"Task  Name: '{task_name}'")
        print(f"Task    ID: '{tid}'")
        print(f"Metric    : {out['eval_metric']}")
        print(f"Test Score: {out['test_score']:.4f}")
        print(f"Val  Score: {out['val_score']:.4f}")
        print(f"Test Error: {out['test_error']:.4f}")
        print(f"Val  Error: {out['val_error']:.4f}")
        print(f"Fit   Time: {out['time_fit']:.3f}s")
        print(f"Infer Time: {out['time_predict']:.3f}s")
