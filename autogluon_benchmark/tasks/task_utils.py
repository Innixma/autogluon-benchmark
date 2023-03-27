import openml
import time

import numpy as np
import pandas as pd

from openml.exceptions import OpenMLServerException

from ..frameworks.autogluon.run import run
from ..utils.data_utils import convert_to_raw

from autogluon.common.savers import save_pd, save_json


def get_task(task_id: int):
    task = openml.tasks.get_task(task_id)
    return task


def get_dataset(task):
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    return X, y


def get_ag_problem_type(task):
    if task.task_type_id.name == 'SUPERVISED_CLASSIFICATION':
        if len(task.class_labels) > 2:
            problem_type = 'multiclass'
        else:
            problem_type = 'binary'
    elif task.task_type_id.name == 'SUPERVISED_REGRESSION':
        problem_type = 'regression'
    else:
        raise AssertionError(f'Unsupported task type: {task.task_type_id.name}')
    return problem_type


def get_task_with_retry(task_id: int, max_delay_exp: int = 8):
    delay_exp = 0
    while True:
        try:
            print(f'Getting task {task_id}')
            task = get_task(task_id=task_id)
            print(f'Got task {task_id}')
            return task
        except OpenMLServerException as e:
            delay = 2 ** delay_exp
            delay_exp += 1
            if delay_exp > max_delay_exp:
                raise ValueError("Unable to get task after 10 retries")
            print(e)
            print(f'Retry in {delay}s...')
            time.sleep(delay)
            continue


def get_task_data(task):
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    return X, y


class TaskWrapper:
    def __init__(self, task):
        self.task = task
        self.X, self.y = get_task_data(task=self.task)
        self.problem_type = get_ag_problem_type(self.task)
        self.label = self.task.target_name

    @classmethod
    def from_task_id(cls, task_id: int):
        task = get_task_with_retry(task_id=task_id)
        return cls(task)

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


def run_task(task, n_folds=None, n_repeats=1, n_samples=1, init_args=None, fit_args=None, print_leaderboard=True):
    if isinstance(task, int):
        task = get_task_with_retry(task)

    problem_type = get_ag_problem_type(task)
    n_repeats_full, n_folds_full, n_samples_full = task.get_split_dimensions()
    if n_folds is None:
        n_folds = n_folds_full
    if n_repeats is None:
        n_repeats = n_repeats_full
    if n_samples is None:
        n_samples = n_samples_full

    X, y = get_task_data(task=task)
    # X = convert_to_raw(X)
    results = []
    if isinstance(n_folds, int):
        n_folds = list(range(n_folds))
    for repeat_idx in range(n_repeats):
        for fold_idx in n_folds:
            for sample_idx in range(n_samples):
                split_info = dict(
                    repeat=repeat_idx,
                    fold=fold_idx,
                    sample=sample_idx,
                )
                train_indices, test_indices = task.get_train_test_split_indices(**split_info)
                X_train = X.loc[train_indices]
                y_train = y[train_indices]
                X_test = X.loc[test_indices]
                y_test = y[test_indices]

                print(
                                    'Repeat #{}, fold #{}, samples {}: X_train.shape: {}, '
                                    'y_train.shape {}, X_test.shape {}, y_test.shape {}'.format(
                                        repeat_idx, fold_idx, sample_idx, X_train.shape, y_train.shape, X_test.shape,
                                        y_test.shape,
                                    )
                                )

                # Save and get_task_dict data to remove any pre-set dtypes, we want to observe performance from worst-case scenario: raw csv

                result = run(
                    X_train=X_train,
                    y_train=y_train,
                    label=task.target_name,
                    X_test=X_test,
                    y_test=y_test,
                    init_args=init_args,
                    fit_args=fit_args,
                    problem_type=problem_type,
                )
                result.update(split_info)
                result['task_id'] = task.task_id
                result['problem_type'] = problem_type
                results.append(result)

    return results


def run_config(task_names, task_metadata, n_folds, config):
    score_dict = dict()
    config = config.copy()
    name = config.pop('name')
    for task_name in task_names:
        task_id = int(task_metadata[task_metadata['name'] == task_name]['tid'].iloc[0])  # openml task id

        result = run_task(task_id, n_folds=n_folds, **config)
        try:
            pass
        except Exception as err:
            score_dict[task_name] = {'is_valid': False, 'exception': err}
            print('Exception Encountered:')
            print(err)
        else:
            score_dict[task_name] = dict(
                is_valid=True,
                result=result,
            )
            score = []
            time_fit = []
            time_predict = []
            for r in result:
                score.append(r['test_score'])
                time_fit.append(r['time_fit'])
                time_predict.append(r['time_predict'])
            score = float(np.mean(score))
            time_fit = float(np.mean(time_fit))
            time_predict = float(np.mean(time_predict))
            print(f'{task_name} score: {round(score, 5)}, time_fit: {round(time_fit, 2)}s, time_predict: {round(time_predict, 4)}s')

    from collections import defaultdict

    df = defaultdict(list)
    cols = ['test_score', 'val_score', 'time_fit', 'time_predict', 'eval_metric', 'test_error', 'fold', 'repeat', 'sample', 'task_id', 'problem_type']
    for task_name, task_result in score_dict.items():
        is_valid = task_result['is_valid']
        if is_valid:
            result = task_result['result']
            for r in result:
                df['name'].append(name)
                df['task_name'].append(task_name)
                for col in cols:
                    df[col].append(r[col])
        else:
            print(f'Task {task_name} failed with exception:')
            print(task_result['exception'])
    df_final = pd.DataFrame(df)
    return df_final


def run_configs(task_names, task_metadata, n_folds, configs):
    df_final = []
    for config in configs:
        df_final.append(run_config(task_names=task_names, task_metadata=task_metadata, n_folds=n_folds, config=config))
    df_final = pd.concat(df_final, ignore_index=True)
    return df_final
