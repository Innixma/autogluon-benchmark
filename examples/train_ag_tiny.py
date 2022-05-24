
import numpy as np
import pandas as pd

from autogluon.common.savers import save_pd

from autogluon_benchmark.tasks import task_loader, task_utils


def run_config(task_names, task_dict, n_folds, config):
    score_dict = dict()
    config = config.copy()
    name = config.pop('name')
    for task_name in task_names:
        task_id = task_dict[task_name]['openml_task_id']  # openml task id

        result = task_utils.run_task(task_id, n_folds=n_folds, **config)
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
    cols = ['test_score', 'time_fit', 'time_predict', 'eval_metric', 'test_error', 'fold', 'repeat', 'sample', 'task_id', 'problem_type']
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


def run_configs(task_names, task_dict, n_folds, configs):
    df_final = []
    for config in configs:
        df_final.append(run_config(task_names=task_names, task_dict=task_dict, n_folds=n_folds, config=config))
    df_final = pd.concat(df_final, ignore_index=True)
    return df_final


if __name__ == "__main__":
    save_path_prefix = 'out/ag_tiny/'
    task_dict = task_loader.get_task_dict(['ag_tiny.yaml'])

    n_folds = [0]
    num_datasets = 10

    print(len(task_dict.keys()))
    task_names = list(task_dict.keys())[:num_datasets]
    print(task_names)

    config1 = dict(
        name='NN_TORCH',
        fit_args={
            'hyperparameters': {'NN_TORCH': {}},
        },
    )
    config2 = dict(
        name='XGB',
        fit_args={
            'hyperparameters': {'XGB': {}},
        }
    )

    configs = [
        config1,
        config2,
    ]

    df_final = run_configs(task_names=task_names, task_dict=task_dict, n_folds=n_folds, configs=configs)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_final)

    save_path_df = save_path_prefix + 'result.csv'
    save_pd.save(path=save_path_df, df=df_final)
