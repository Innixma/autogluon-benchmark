import openml
from autogluon.task import TabularPrediction as ag_task
from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd


def run_ag(X_train, y_train, X_test, y_test, label, ag_args=None):
    if ag_args is None:
        ag_args = {}
    X_train[label] = y_train
    X_test[label] = y_test

    X_train = X_train.copy()
    X_test = X_test.copy()

    # print(X_train)
    # print(X_test)

    predictor = ag_task.fit(
        train_data=X_train,
        label=label,
        **ag_args,
    )

    X_test = X_test.reset_index(drop=True)  # TODO: FIXME SO THIS ISN'T NECESSARY!
    # predictor.leaderboard(X_test)
    return predictor


def get_task(task_id: int):
    task = openml.tasks.get_task(task_id)
    return task


def get_dataset(task):
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    return X, y


def run_task(task, n_folds=None, n_repeats=1, n_samples=1, ag_args=None):
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
    X, y = convert_to_raw(X, y)
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

                # Save and load data to remove any pre-set dtypes, we want to observe performance from worst-case scenario: raw csv
                predictor = run_ag(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, label=task.target_name, ag_args=ag_args)
                predictors.append(predictor)
                X_test[task.target_name] = y_test
                scores.append(predictor.evaluate(X_test))

    return predictors, scores


# Remove custom type information
def convert_to_raw(X, y=None):
    train_path = 'tmp/openml/tmp.csv'
    if y is not None:
        X['__TMP_LABEL__'] = y
    save_pd.save(path=train_path, df=X)
    X = load_pd.load(path=train_path)
    if y is not None:
        y = X['__TMP_LABEL__']
        X.drop('__TMP_LABEL__', axis=1, inplace=True)
    return X, y
