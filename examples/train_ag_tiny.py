import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd

from autogluon_benchmark.evaluation.evaluate_utils import compare_frameworks
from autogluon_benchmark.tasks import task_runner
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata


if __name__ == "__main__":
    save_path_prefix = 'out/ag_tiny/'
    task_metadata = load_task_metadata('task_metadata_244.csv')
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

    df_final = task_runner.run_configs(task_names=task_names, task_metadata=task_metadata, n_folds=n_folds, configs=configs)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_final)

    save_path_df = save_path_prefix + 'result.csv'
    save_pd.save(path=save_path_df, df=df_final)

    df_renamed = df_final.rename(columns=dict(
        name='framework',
        task_name='dataset',
        time_fit='time_train_s',
        time_predict='time_infer_s',
        test_error='metric_error',
    ))
    out = compare_frameworks(
        results_raw=df_renamed,
        columns_to_agg_extra=['time_infer_s']
    )
