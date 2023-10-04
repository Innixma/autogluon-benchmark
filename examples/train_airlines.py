
import numpy as np

from autogluon_benchmark.tasks import task_loader, task_runner


if __name__ == "__main__":
    task_dict = task_loader.get_task_dict()
    task_name = 'airlines'  # task name in yaml config
    task_id = task_dict[task_name]['openml_task_id']  # openml task id
    n_folds = 1  # do 5 folds of train/val split

    init_args = {
        'eval_metric': 'roc_auc',
    }

    fit_args = {
        'time_limit': 1500,
        # 'use_bag_holdout': True,
        'hyperparameters': {
            'KNN': {},
            'RF': {},
            'GBM': {},
        },
        'num_bag_folds': 5,
        'num_stack_levels': 1,
        'num_bag_sets': 1,
        'verbosity': 2,
    }

    predictors, scores = task_runner.run_task(task_id, n_folds=n_folds, init_args=init_args, fit_args=fit_args)
    score = float(np.mean(scores))
    if len(scores) > 1:
        score_std = np.std(scores, ddof=1)
    else:
        score_std = 0.0  # Should this be np.inf?
    print(f'{task_name} score: {round(score, 5)} (+- {round(score_std, 5)})')
