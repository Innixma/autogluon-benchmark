import pandas as pd
from openml.tasks.task import OpenMLSupervisedTask

from autogluon.common.savers import save_pd, save_json

from .task_utils import get_task_data, get_ag_problem_type, get_task_with_retry
from ..frameworks.autogluon.run import run


class OpenMLTaskWrapper:
    def __init__(self, task: OpenMLSupervisedTask):
        assert isinstance(task, OpenMLSupervisedTask)
        self.task: OpenMLSupervisedTask = task
        self.X, self.y = get_task_data(task=self.task)
        self.problem_type = get_ag_problem_type(self.task)
        self.label = self.task.target_name

    @classmethod
    def from_task_id(cls, task_id: int):
        task = get_task_with_retry(task_id=task_id)
        return cls(task)

    @property
    def task_id(self) -> int:
        return self.task.task_id

    @property
    def dataset_id(self) -> int:
        return self.task.dataset_id

    def get_split_dimensions(self):
        n_repeats, n_folds, n_samples = self.task.get_split_dimensions()
        return n_repeats, n_folds, n_samples

    def combine_X_y(self):
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

    def get_split_indices(self, repeat=0, fold=0, sample=0):
        train_indices, test_indices = self.task.get_train_test_split_indices(repeat=repeat, fold=fold, sample=sample)
        return train_indices, test_indices

    def get_train_test_split(self, repeat=0, fold=0, sample=0, train_indices=None, test_indices=None):
        if train_indices is None or test_indices is None:
            train_indices, test_indices = self.task.get_train_test_split_indices(repeat=repeat, fold=fold, sample=sample)
        X_train = self.X.loc[train_indices]
        y_train = self.y[train_indices]
        X_test = self.X.loc[test_indices]
        y_test = self.y[test_indices]
        return X_train, y_train, X_test, y_test

    def get_train_test_split_combined(self, repeat=0, fold=0, sample=0, train_indices=None, test_indices=None):
        X_train, y_train, X_test, y_test = self.get_train_test_split(
            repeat=repeat,
            fold=fold,
            sample=sample,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        train_data = pd.concat([X_train, y_train.to_frame(name=self.label)], axis=1)
        test_data = pd.concat([X_test, y_test.to_frame(name=self.label)], axis=1)
        return train_data, test_data


class AutoGluonTaskWrapper(OpenMLTaskWrapper):
    def run(
            self,
            repeat: int = 0,
            fold: int = 0,
            sample: int = 0,
            init_args: dict = None,
            fit_args: dict = None,
            extra_kwargs: dict = None) -> dict:
        X_train, y_train, X_test, y_test = self.get_train_test_split(repeat=repeat, fold=fold, sample=sample)
        out = run(X_train=X_train, y_train=y_train, label=self.label, X_test=X_test, y_test=y_test,
                  init_args=init_args, fit_args=fit_args, extra_kwargs=extra_kwargs, problem_type=self.problem_type)
        out["tid"] = self.task_id
        out["problem_type"] = self.problem_type
        out["repeat"] = repeat
        out["fold"] = fold
        out["sample"] = sample
        return out
