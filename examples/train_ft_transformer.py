from typing import Callable, List
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager

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


@contextmanager
def catchtime(name: str, logger = None) -> float:
    start = perf_counter()
    print_fun = print if logger is None else logger.info
    try:
        print_fun(f"start: {name}")
        yield lambda: perf_counter() - start
    finally:
        print_fun(f"Time for {name}: {perf_counter() - start:.4f} secs")


def cache_function_dataframe(
        fun: Callable[[], pd.DataFrame],
        cache_name: str,
        ignore_cache: bool = False,
        cache_path: Path = None
):
    f"""
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv.zip`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :return: result of fun()
    """
    cache_file = Path(cache_path) / (cache_name + ".csv")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        return pd.read_csv(cache_file)
    else:
        print(f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file")
        with catchtime("Evaluate function."):
            df = fun()
            assert isinstance(df, pd.DataFrame)
            df.to_csv(cache_file, index=False)
            return pd.read_csv(cache_file)


@dataclass
class Experiment:
    expname: str  # name of the parent experiment used to store the file
    name: str  # name of the specific experiment, e.g. "localsearch"
    run_fun: Callable[[], list]  # function to execute to obtain results

    def data(self, ignore_cache: bool = False):
        return cache_function_dataframe(
            lambda: pd.DataFrame(self.run_fun()),
            cache_name=self.name,
            ignore_cache=ignore_cache,
            cache_path=self.expname,
        )


def fit_ag(task: AutoGluonTaskWrapper, fold: int, task_name: str, fit_args: dict):
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

    df_results = pd.DataFrame([out])
    ordered_columns = ["dataset", "fold", "framework", "test_error", "val_error", "eval_metric", "test_score", "val_score", "time_fit"]
    columns_reorder = ordered_columns + [c for c in df_results.columns if c not in ordered_columns]
    df_results = df_results[columns_reorder]
    return df_results


if __name__ == '__main__':
    out_dir = "./results_bingzhao2"
    ignore_cache = False

    task_metadata = load_task_metadata('task_metadata.csv')
    task_metadata_tiny = task_metadata[task_metadata['NumberOfInstances'] <= 2000]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfFeatures'] <= 100]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfClasses'] <= 10]
    tids = task_metadata_tiny["tid"].to_list()
    print(tids)

    # tids = tids[:2]  # TODO: This is for demonstration purposes, comment this out to train on more datasets
    folds = [0, 1, 2, 3, 4]  # How many folds ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] is all folds), more folds = less noise in results

    frameworks_dict = dict(
        FTT={
            "hyperparameters": {
                "FT_TRANSFORMER": {  # # FT_TRANSFORMER hyperparameters go here

                },
            },
        },
        FTT_v2={
            "hyperparameters": {
                "FT_TRANSFORMER": {  # # FT_TRANSFORMER hyperparameters go here
                    "env.per_gpu_batch_size": 128,
                },
            },
        },
        FTT_v3={
            "hyperparameters": {
                "FT_TRANSFORMER": {  # # FT_TRANSFORMER hyperparameters go here
                    "env.per_gpu_batch_size": 128,
                    "pretrainer": True,
                    "pretrainer.augmentation_type": "permutation",
                    "pretrainer.corruption_rate": 0.6,
                    "pretrainer.objective": "reconstruction",
                    "pretrainer.start_pretrain_coefficient": 1,
                    "pretrainer.end_pretrain_coefficient": 0.03,
                    "pretrainer.pretrain_epochs": 5,
                    "pretrainer.decay_pretrain_coefficient": 0.5,
                    "pretrainer.temperature": 1,
                    "model.fusion_transformer.row_attention": True,
                    "model.fusion_transformer.global_token": True,
                    "model.fusion_transformer.row_attention_layer": "last",
                    "optimization.row_attention_weight_decay": 0.1,
                    "env.test_ensemble_rounds": 10,
                },
            },
        },
        FTT_v4={
            "hyperparameters": {
                "FT_TRANSFORMER": {  # # FT_TRANSFORMER hyperparameters go here
                    "env.per_gpu_batch_size": 128,
                    "pretrainer": True,
                    "pretrainer.augmentation_type": "permutation",
                    "pretrainer.corruption_rate": 0.6,
                    "pretrainer.objective": "self_distill",
                    "pretrainer.start_pretrain_coefficient": 0.1,
                    "pretrainer.end_pretrain_coefficient": 0.1,
                    "pretrainer.pretrain_epochs": 0,
                    "pretrainer.decay_pretrain_coefficient": 0.6,
                    "pretrainer.temperature": 1,
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
        CAT={  # CatBoost
            "hyperparameters": {
                "CAT": {},
            },
        },
        XGB={  # XGBoost
            "hyperparameters": {
                "XGB": {},
            },
        },
        NN_TORCH={  # Torch Neural Network
            "hyperparameters": {
                "NN_TORCH": {},
            },
        },
        RF={  # Random Forest
            "hyperparameters": {
                "RF": {},
            },
        },
        XT={  # Extra Trees
            "hyperparameters": {
                "XT": {},
            },
        },
    )
    shared_args = dict(
        time_limit=3600,
        fit_weighted_ensemble=False,
    )
    for key in frameworks_dict:
        frameworks_dict[key].update(shared_args)

    frameworks = [
        # "FTT",
        "FTT_v2",
        "FTT_v3",
        "FTT_v4",
        "GBM",
        "XGB",
        "CAT",
        "FASTAI",
        "NN_TORCH",
        "RF",
        "XT",
    ]

    result_lst = []
    for tid in tids:
        task = AutoGluonTaskWrapper.from_task_id(task_id=tid)
        task_name = task_metadata[task_metadata["tid"] == tid]["name"].iloc[0]
        for fold in folds:
            for framework in frameworks:
                cache_name = f"{tid}/{fold}/{framework}"
                fit_args = frameworks_dict[framework]
                experiment = Experiment(
                    expname=out_dir, name=cache_name,
                    run_fun=lambda: fit_ag(task=task, fold=fold, task_name=task_name, fit_args=fit_args),
                )
                out = experiment.data(ignore_cache=ignore_cache)
                result_lst.append(out)

    df_results = pd.concat(result_lst, ignore_index=True)
    df_results = df_results.rename(columns=dict(
        time_fit="time_train_s",
        time_predict="time_infer_s",
        test_error="metric_error",
    ))

    save_pd.save(path=f"{out_dir}/results.csv", df=df_results)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_results)

    results_ranked, results_ranked_by_dataset = compare_frameworks(
        results_raw=df_results,
        columns_to_agg_extra=["time_infer_s"],
    )
    save_pd.save(path=f"{out_dir}/results_ranked.csv", df=results_ranked)
    save_pd.save(path=f"{out_dir}/results_ranked_by_dataset.csv", df=results_ranked_by_dataset)
