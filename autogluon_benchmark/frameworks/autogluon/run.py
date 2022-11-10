
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
           time_compile=None,
           test_score=None,
           test_error=None,
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


def run(X_train, y_train, label: str, X_test, y_test, init_args: dict = None, fit_args: dict = None, compile_args: dict = None, extra_kwargs: dict = None, problem_type=None):
    if init_args is None:
        init_args = {}
    if fit_args is None:
        fit_args = {}
    if compile_args is None:
        compile_args = {}
    if extra_kwargs is None:
        extra_kwargs = {}
    if problem_type is not None:
        init_args['problem_type'] = problem_type
    if 'eval_metric' not in init_args:
        if init_args.get('problem_type', None):
            init_args['eval_metric'] = ag_eval_metric_map[init_args['problem_type']]
    X_train = pd.concat([X_train, y_train.to_frame(name=label)], axis=1)

    with Timer() as timer_fit:
        predictor = TabularPredictor(label=label, **init_args).fit(
            train_data=X_train,
            **fit_args,
        )

    is_classification = predictor.problem_type in ['binary', 'multiclass']

    with Timer() as timer_compile:
        if len(compile_args) > 0:
            compiler_configs = compile_args['compiler_configs']
            predictor.compile_models('best', compiler_configs=compiler_configs)
    predictor.persist_models('best')

    if is_classification:
        with Timer() as timer_predict:
            probabilities = predictor.predict_proba(X_test, as_multiclass=True)
        predictions = probabilities.idxmax(axis=1).to_numpy()
    else:
        with Timer() as timer_predict:
            predictions = predictor.predict(X_test, as_pandas=False)
        probabilities = None

    # X_test = X_test.copy()
    # X_test[label] = y_test
    X_test = pd.concat([X_test, y_test.to_frame(name=label)], axis=1)

    eval_metric = predictor.eval_metric.name
    test_score = predictor.evaluate(X_test, silent=True, auxiliary_metrics=False)[eval_metric]
    test_error = predictor.eval_metric._optimum - test_score

    _leaderboard_extra_info = extra_kwargs.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = extra_kwargs.get('_leaderboard_test', True)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = X_test

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(leaderboard)

    models_count = len(leaderboard)

    predictor.unpersist_models('all')

    return result(
        predictions=predictions,
        probabilities=probabilities,
        truth=y_test,
        models_count=models_count,
        time_fit=timer_fit.duration,
        time_predict=timer_predict.duration,
        time_compile=timer_compile.duration,
        # extra
        test_score=test_score,
        test_error=test_error,
        eval_metric=eval_metric,
        predictor=predictor,
        leaderboard=leaderboard,
    )
