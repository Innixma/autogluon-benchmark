
import numpy as np
import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd

from autogluon_benchmark.evaluation.evaluate_utils import compare_frameworks
from autogluon_benchmark.tasks import task_utils
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata


def run_config(task_names, task_metadata, n_folds, config):
    score_dict = dict()
    config = config.copy()
    name = config.pop('name')
    for task_name in task_names:
        task_id = int(task_metadata[task_metadata['name'] == task_name]['tid'].iloc[0])  # openml task id

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
    cols = ['test_score', 'val_score', 'time_fit', 'time_predict', 'eval_metric', 'test_error', 'fold', 'repeat', 'sample', 'task_id', 'problem_type']
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


def run_configs(task_names, task_metadata, n_folds, configs):
    df_final = []
    for config in configs:
        df_final.append(run_config(task_names=task_names, task_metadata=task_metadata, n_folds=n_folds, config=config))
    df_final = pd.concat(df_final, ignore_index=True)
    return df_final


if __name__ == "__main__":
    save_path_prefix = 'out/ag_tiny/'
    task_metadata = load_task_metadata('task_metadata_289.csv')
    task_metadata = task_metadata.drop_duplicates(subset=['name'])
    task_metadata_tiny = task_metadata[task_metadata['NumberOfInstances'] <= 1000]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfFeatures'] <= 100]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['NumberOfClasses'] <= 10]
    task_metadata_tiny = task_metadata_tiny[task_metadata_tiny['task_type'] == 'Supervised Classification']

    task_names = list(task_metadata_tiny['name'])

    print(task_metadata_tiny)

    n_folds = [0]
    num_datasets = 10
    task_names = task_names[:num_datasets]
    # task_names = ['synthetic_control']
    # task_names = ['monks-problems-1']
    # task_names = ['irish']

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

    df_final = run_configs(task_names=task_names, task_metadata=task_metadata_tiny, n_folds=n_folds, configs=configs)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_final)

    save_path_df = save_path_prefix + 'result.csv'
    save_pd.save(path=save_path_df, df=df_final)

    df_renamed = df_final.rename(columns=dict(
        name='framework',
        task_name='dataset',
        time_fit='time_train_s',
        test_error='metric_error',
    ))
    out = compare_frameworks(results_raw=df_renamed)
