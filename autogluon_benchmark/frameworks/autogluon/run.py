
from autogluon.tabular import TabularPrediction as ag_task


def run(X_train, y_train, label: str, fit_args: dict = None):
    if fit_args is None:
        fit_args = {}
    X_train[label] = y_train

    predictor = ag_task.fit(
        train_data=X_train,
        label=label,
        **fit_args,
    )

    return predictor
