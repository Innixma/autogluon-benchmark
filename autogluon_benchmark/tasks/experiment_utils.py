from __future__ import annotations

from typing import Callable, List
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager

import pandas as pd

from .task_wrapper import AutoGluonTaskWrapper


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
    cache_path: Path | str,
    ignore_cache: bool = False,
) -> pd.DataFrame:
    f"""
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv`
    :param cache_path: folder where to write cache files
    :param ignore_cache: whether to recompute even if the cache is present
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
            cache_path=self.expname,
            ignore_cache=ignore_cache,
        )


# TODO: Cleanup this code
def fit_ag(task: AutoGluonTaskWrapper, fold: int, task_name: str, fit_args: dict, method: str, verbose: bool = False):
    print(f"Running Task Name: '{task_name}' on fold {fold} with method '{method}'")
    out = task.run(fold=fold, fit_args=fit_args)

    out["framework"] = method
    out["dataset"] = task_name
    if verbose:
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


# TODO: Cleanup this code, make it into a class?
def run_experiments(
    *,
    out_dir: str,
    tids: List[int],
    folds: List[int],
    methods: List[str],
    methods_dict: dict,
    task_metadata: pd.DataFrame,
    ignore_cache: bool,
):
    dataset_names = [task_metadata[task_metadata["tid"] == tid]["name"].iloc[0] for tid in tids]
    print(
        f"Fitting {len(tids)} datasets and {len(folds)} folds for a total of {len(tids) * len(folds)} tasks"
        f"\n\tFitting {len(methods)} methods on {len(tids) * len(folds)} tasks for a total of {len(tids) * len(folds) * len(methods)} jobs..."
        f"\n\tTIDs    : {tids}"
        f"\n\tDatasets: {dataset_names}"
        f"\n\tFolds   : {folds}"
        f"\n\tMethods : {methods}"
    )
    result_lst = []
    for tid in tids:
        task = AutoGluonTaskWrapper.from_task_id(task_id=tid)
        task_name = task_metadata[task_metadata["tid"] == tid]["name"].iloc[0]
        for fold in folds:
            for method in methods:
                cache_name = f"data/tasks/{tid}/{fold}/{method}/results"
                fit_args = methods_dict[method]
                experiment = Experiment(
                    expname=out_dir, name=cache_name,
                    run_fun=lambda: fit_ag(task=task, fold=fold, task_name=task_name, fit_args=fit_args, method=method),
                )
                out = experiment.data(ignore_cache=ignore_cache)
                result_lst.append(out)

    df_results = pd.concat(result_lst, ignore_index=True)
    return df_results
