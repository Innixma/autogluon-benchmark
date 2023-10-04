import numpy as np
import pandas as pd

from .task_wrapper import TaskWrapper
from ..frameworks.autogluon.run import run


def run_task(task, n_folds=None, n_repeats=1, n_samples=1, init_args=None, fit_args=None, print_leaderboard=True):
    if isinstance(task, int):
        task_wrapper = TaskWrapper.from_task_id(task_id=task)
    else:
        task_wrapper = TaskWrapper(task=task)

    n_repeats_full, n_folds_full, n_samples_full = task_wrapper.get_split_dimensions()
    if n_folds is None:
        n_folds = n_folds_full
    if n_repeats is None:
        n_repeats = n_repeats_full
    if n_samples is None:
        n_samples = n_samples_full

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
                X_train, y_train, X_test, y_test = task_wrapper.get_train_test_split(**split_info)

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
                    label=task_wrapper.label,
                    X_test=X_test,
                    y_test=y_test,
                    init_args=init_args,
                    fit_args=fit_args,
                    problem_type=task_wrapper.problem_type,
                )
                result.update(split_info)
                result['task_id'] = task_wrapper.task_id
                result['problem_type'] = task_wrapper.problem_type
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
