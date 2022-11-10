import openml
import time

from openml.exceptions import OpenMLServerException

from ..frameworks.autogluon.run import run
from ..utils.data_utils import convert_to_raw


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


def run_task(task, n_folds=None, n_repeats=1, n_samples=1, init_args=None, fit_args=None, compile_args=None, print_leaderboard=True):
    if isinstance(task, int):
        task_id = task
        delay_exp = 0
        while True:
            try:
                print(f'Getting task {task_id}')
                task = openml.tasks.get_task(task_id)
                print(f'Got task {task_id}')
            except OpenMLServerException as e:
                delay = 2 ** delay_exp
                delay_exp += 1
                if delay_exp > 10:
                    raise ValueError("Unable to get task after 10 retries")
                print(e)
                print(f'Retry in {delay}s...')
                time.sleep(delay)
                continue
            break

    problem_type = get_ag_problem_type(task)
    n_repeats_full, n_folds_full, n_samples_full = task.get_split_dimensions()
    if n_folds is None:
        n_folds = n_folds_full
    if n_repeats is None:
        n_repeats = n_repeats_full
    if n_samples is None:
        n_samples = n_samples_full

    X, y, _, _ = task.get_dataset().get_data(task.target_name)
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
                    compile_args=compile_args,
                    problem_type=problem_type,
                )
                result.update(split_info)
                result['task_id'] = task.task_id
                result['problem_type'] = problem_type
                results.append(result)

    return results
