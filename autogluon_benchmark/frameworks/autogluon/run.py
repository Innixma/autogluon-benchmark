
import pandas as pd

from autogluon.tabular import TabularPredictor

from ...utils.time_utils import Timer


def result(*,
           predictions=None,
           probabilities=None,
           truth=None,
           models_count=None,
           time_fit=None,
           time_predict=None,
           test_score=None,
           val_score=None,
           test_error=None,
           val_error=None,
           eval_metric=None,
           **others):
    return locals()


ag_problem_type_map = {
    'Supervised Regression': 'regression',
}

ag_eval_metric_map = {
    'binary': 'roc_auc',
    'multiclass': 'log_loss',
    'regression': 'rmse',
}


def run(X_train, y_train, label: str, X_test, y_test, init_args: dict = None, fit_args: dict = None, extra_kwargs: dict = None, problem_type=None):
    if init_args is None:
        init_args = {}
    if fit_args is None:
        fit_args = {}
    if extra_kwargs is None:
        extra_kwargs = {}
    if problem_type is not None:
        init_args['problem_type'] = problem_type
    if 'eval_metric' not in init_args:
        if init_args.get('problem_type', None):
            init_args['eval_metric'] = ag_eval_metric_map[init_args['problem_type']]
    X_train = pd.concat([X_train, y_train.to_frame(name=label)], axis=1)

    with Timer() as timer_fit:
        predictor: TabularPredictor = TabularPredictor(label=label, **init_args).fit(
            train_data=X_train,
            **fit_args,
        )

    is_classification = predictor.problem_type in ['binary', 'multiclass']

    predictor.persist('best')

    if is_classification:
        with Timer() as timer_predict:
            probabilities = predictor.predict_proba(X_test, as_multiclass=True)
        predictions = predictor.predict_from_proba(probabilities)
    else:
        with Timer() as timer_predict:
            predictions = predictor.predict(X_test)
        probabilities = None

    eval_metric = predictor.eval_metric.name
    if probabilities is None:
        test_score = predictor.evaluate_predictions(y_true=y_test, y_pred=predictions, auxiliary_metrics=False)[eval_metric]
    else:
        test_score = predictor.evaluate_predictions(y_true=y_test, y_pred=probabilities, auxiliary_metrics=False)[eval_metric]
    test_error = predictor.eval_metric.convert_score_to_error(test_score)

    _leaderboard_extra_info = extra_kwargs.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = extra_kwargs.get('_leaderboard_test', True)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        test_data = pd.concat([X_test, y_test.to_frame(name=label)], axis=1)
        leaderboard_kwargs['data'] = test_data

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(leaderboard)
    val_score = leaderboard[leaderboard['model'] == predictor.model_best]['score_val'].iloc[0]
    val_error = predictor.eval_metric.convert_score_to_error(val_score)

    models_count = len(leaderboard)

    predictor.unpersist()

    return result(
        predictions=predictions,
        probabilities=probabilities,
        truth=y_test,
        models_count=models_count,
        time_fit=timer_fit.duration,
        time_predict=timer_predict.duration,
        # extra
        test_score=test_score,
        val_score=val_score,
        test_error=test_error,
        val_error=val_error,
        eval_metric=eval_metric,
        predictor=predictor,
        leaderboard=leaderboard,
    )
