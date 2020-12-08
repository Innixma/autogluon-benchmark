import openml

from ..frameworks.autogluon.run import run
from ..utils.data_utils import convert_to_raw


def get_task(task_id: int):
    task = openml.tasks.get_task(task_id)
    return task


def get_dataset(task):
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    return X, y


def run_task(task, n_folds=None, n_repeats=1, n_samples=1, fit_args=None, print_leaderboard=True):
    if isinstance(task, int):
        task = openml.tasks.get_task(task)

    n_repeats_full, n_folds_full, n_samples_full = task.get_split_dimensions()
    if n_folds is None:
        n_folds = n_folds_full
    if n_repeats is None:
        n_repeats = n_repeats_full
    if n_samples is None:
        n_samples = n_samples_full

    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    X = convert_to_raw(X)
    predictors = []
    scores = []
    for repeat_idx in range(n_repeats):
        for fold_idx in range(n_folds):
            for sample_idx in range(n_samples):
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=repeat_idx,
                    fold=fold_idx,
                    sample=sample_idx,
                )
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
                predictor = run(X_train=X_train, y_train=y_train, label=task.target_name, fit_args=fit_args)
                predictors.append(predictor)
                X_test[task.target_name] = y_test
                scores.append(predictor.evaluate(X_test))
                if print_leaderboard:
                    predictor.leaderboard(X_test)

    return predictors, scores
