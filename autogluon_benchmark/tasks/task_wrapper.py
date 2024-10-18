from __future__ import annotations

import numpy as np
import pandas as pd
from openml.tasks.task import OpenMLSupervisedTask
from typing_extensions import Self

from autogluon.common.savers import save_pd, save_json
from autogluon.core.utils import generate_train_test_split

from .task_utils import get_task_data, get_ag_problem_type, get_task_with_retry
from ..frameworks.autogluon.run import run
from ..metadata.metadata_loader import load_task_metadata


class OpenMLTaskWrapper:
    def __init__(self, task: OpenMLSupervisedTask):
        assert isinstance(task, OpenMLSupervisedTask)
        self.task: OpenMLSupervisedTask = task
        self.X, self.y = get_task_data(task=self.task)
        self.problem_type = get_ag_problem_type(self.task)
        self.label = self.task.target_name

    @classmethod
    def from_task_id(cls, task_id: int) -> Self:
        task = get_task_with_retry(task_id=task_id)
        return cls(task)

    @classmethod
    def from_name(cls, dataset: str, task_metadata: pd.DataFrame = None) -> Self:
        if task_metadata is None:
            task_metadata = load_task_metadata()
        assert "name" in task_metadata, f"`name` column missing in task_metadata! Columns: {list(task_metadata.columns)}"
        assert "tid" in task_metadata, f"`tid` column missing in task_metadata! Columns: {list(task_metadata.columns)}"
        assert task_metadata.value_counts("name").max() == 1, (f"Duplicate names found in task_metadata! "
                                                               f"This shouldn't occur: {task_metadata.value_counts('name')}")
        task_id = task_metadata.set_index("name").loc[dataset, "tid"]
        return cls.from_task_id(task_id=task_id)

    @property
    def task_id(self) -> int:
        return self.task.task_id

    @property
    def dataset_id(self) -> int:
        return self.task.dataset_id

    def get_split_dimensions(self) -> tuple[int, int, int]:
        n_repeats, n_folds, n_samples = self.task.get_split_dimensions()
        return n_repeats, n_folds, n_samples

    def combine_X_y(self) -> pd.DataFrame:
        return pd.concat([self.X, self.y.to_frame(name=self.label)], axis=1)

    def save_data(self, path: str, file_type='.csv', train_indices=None, test_indices=None):
        data = self.combine_X_y()
        if train_indices is not None and test_indices is not None:
            train_data = data.loc[train_indices]
            test_data = data.loc[test_indices]
            save_pd.save(f"{path}train{file_type}", train_data)
            save_pd.save(f"{path}test{file_type}", test_data)
        else:
            save_pd.save(f"{path}data{file_type}", data)

    def save_metadata(self, path: str):
        metadata = dict(
            label=self.label,
            problem_type=self.problem_type,
            num_rows=len(self.X),
            num_cols=len(self.X.columns),
            task_id=self.task.task_id,
            dataset_id=self.task.dataset_id,
            openml_url=self.task.openml_url,
        )
        path_full = f"{path}metadata.json"
        save_json.save(path=path_full, obj=metadata)

    def get_split_indices(self, fold: int = 0, repeat: int = 0, sample: int = 0) -> tuple[np.ndarray, np.ndarray]:
        train_indices, test_indices = self.task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=sample)
        return train_indices, test_indices

    def get_train_test_split(
        self,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_size: int | float = None,
        test_size: int | float = None,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if train_indices is None or test_indices is None:
            train_indices, test_indices = self.get_split_indices(fold=fold, repeat=repeat, sample=sample)
        X_train = self.X.loc[train_indices]
        y_train = self.y[train_indices]
        X_test = self.X.loc[test_indices]
        y_test = self.y[test_indices]

        if train_size is not None:
            X_train, y_train = self.subsample(X=X_train, y=y_train, size=train_size, random_state=random_state)
        if test_size is not None:
            X_test, y_test = self.subsample(X=X_test, y=y_test, size=test_size, random_state=random_state)

        return X_train, y_train, X_test, y_test

    def subsample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        size: int | float,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if isinstance(size, int) and size >= len(X):
            return X, y
        if isinstance(size, float) and size >= 1:
            return X, y
        X, _, y, _ = generate_train_test_split(
            X=X, y=y, problem_type=self.problem_type, train_size=size, random_state=random_state
        )
        return X, y

    def get_train_test_split_combined(
        self,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_size: int | float = None,
        test_size: int | float = None,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train, y_train, X_test, y_test = self.get_train_test_split(
            fold=fold,
            repeat=repeat,
            sample=sample,
            train_indices=train_indices,
            test_indices=test_indices,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        train_data = pd.concat([X_train, y_train.to_frame(name=self.label)], axis=1)
        test_data = pd.concat([X_test, y_test.to_frame(name=self.label)], axis=1)
        return train_data, test_data

    def subsample_combined(
        self,
        data: pd.DataFrame,
        size: int | float,
        random_state: int = 0,
    ) -> pd.DataFrame:
        data, _ = self.subsample(X=data, y=data[self.label], size=size, random_state=random_state)
        return data


class AutoGluonTaskWrapper(OpenMLTaskWrapper):
    def run(
        self,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
        init_args: dict = None,
        fit_args: dict = None,
        extra_kwargs: dict = None,
    ) -> dict:
        X_train, y_train, X_test, y_test = self.get_train_test_split(fold=fold, repeat=repeat, sample=sample)
        out = run(X_train=X_train, y_train=y_train, label=self.label, X_test=X_test, y_test=y_test,
                  init_args=init_args, fit_args=fit_args, extra_kwargs=extra_kwargs, problem_type=self.problem_type)
        out["tid"] = self.task_id
        out["problem_type"] = self.problem_type
        out["repeat"] = repeat
        out["fold"] = fold
        out["sample"] = sample
        return out

    def out_to_amlb_results(self, out: dict, framework: str, dataset: str) -> pd.DataFrame:
        results = out.copy()
        results["framework"] = framework
        results["dataset"] = dataset
        results.pop("predictions")
        results.pop("probabilities")
        results.pop("truth")
        results.pop("others")
        df_results = pd.DataFrame([results])
        ordered_columns = ["dataset", "fold", "framework", "test_error", "val_error", "eval_metric", "test_score", "val_score", "time_fit"]
        columns_reorder = ordered_columns + [c for c in df_results.columns if c not in ordered_columns]
        df_results = df_results[columns_reorder]
        return df_results
