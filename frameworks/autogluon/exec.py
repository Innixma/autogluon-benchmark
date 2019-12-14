import logging

import pandas as pd
import numpy as np
from autogluon.task.tabular_prediction.tabular_prediction import TabularPrediction as task
from autogluon.utils.tabular.utils.loaders import load_pd
from autogluon.utils.tabular.utils.savers import save_pd

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Timer

import autogluon.utils.tabular.metrics as metrics

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoGluon ****\n")

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'

    X_train = dataset.train.X
    y_train = dataset.train.y
    X_test = dataset.test.X
    y_test = dataset.test.y

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    # Save and load data to remove any pre-set dtypes, we want to observe performance from worst-case scenario: raw csv
    save_pd.save(path='tmp/tmp_file.csv', df=X_train)
    X_train = load_pd.load(path='tmp/tmp_file.csv')
    save_pd.save(path='tmp/tmp_file.csv', df=X_test)
    X_test = load_pd.load(path='tmp/tmp_file.csv')

    X_train['__label__'] = y_train

    with Timer() as training:
        predictor = task.fit(
            train_data=X_train,
            label='__label__',
            output_directory='tmp/',
            time_limits=config.max_runtime_seconds,
            hyperparameter_tune=False,
            eval_metric=perf_metric,
            num_bagging_folds=10,
            stack_ensemble_levels=1,
            verbosity=3,
        )
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None

    if (is_classification) & (len(probabilities.shape) == 1):
        probabilities = np.array([[1-row, row] for row in probabilities])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(predictor._learner.leaderboard(X_test, y_test, silent=True))

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=False)

    return dict(
        models_count=len(predictor._trainer.model_names),
        training_duration=training.duration
    )
