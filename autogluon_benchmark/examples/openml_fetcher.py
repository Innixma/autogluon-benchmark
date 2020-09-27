import openml
from autogluon.task import TabularPrediction as ag_task
from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd
from autogluon_benchmark.tasks import task_utils
import numpy as np


def run_ag(X_train, y_train, X_test, y_test, label, eval_metric):

    train_path = 'tmp/openml/train.csv'
    test_path = 'tmp/openml/test.csv'
    X_train[label] = y_train
    save_pd.save(path=train_path, df=X_train)
    X_test[label] = y_test
    save_pd.save(path=test_path, df=X_test)
    del X_train
    del X_test

    # Save and load data to remove any pre-set dtypes, we want to observe performance from worst-case scenario: raw csv
    X_train = load_pd.load(path=train_path)

    predictor = ag_task.fit(train_data=X_train, label=label, eval_metric=eval_metric)

    X_test = load_pd.load(path=test_path)

    predictor.leaderboard(X_test)


if __name__ == "__main__":

    ag_args = {
        'eval_metric': 'roc_auc',
        'verbosity': 1,
    }

    # task_id = 10101
    # task_id = 7592
    # task_id = 146818
    task_id = 168911
    n_folds=5

    predictors, scores = task_utils.run_task(task_id, n_folds=n_folds, ag_args=ag_args)
    score1 = np.mean(scores)

    ag_args = {
        'eval_metric': 'roc_auc',
        'stopping_metric': 'roc_auc',
        'verbosity': 1,
    }

    predictors, scores = task_utils.run_task(task_id, n_folds=n_folds, ag_args=ag_args)
    score2 = np.mean(scores)

    print('Score1:', score1)
    print('Score2:', score2)

    error1 = 1 - score1
    error2 = 1 - score2

    if error1 == error2:
        diffperc = 0
    if error1 > error2:
        diffperc = 1 - error2 / error1
    else:
        diffperc = -(1 - error1 / error2)
    print('% diff:', diffperc*100)
