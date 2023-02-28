import argparse

import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.loaders.load_s3 import list_bucket_prefix_suffix_contains_s3
from autogluon.common.utils import s3_utils
from autogluon.common.savers import save_pd

import ray


@ray.remote
def get_single_ray(result_path, infer_speed_path):
    return get_single(result_path, infer_speed_path)


def select_ray(data_path_dict: dict):
    print('starting ray...')
    if not ray.is_initialized():
        ray.init()
    # Create and execute all tasks in parallel
    results = []
    for data_paths in data_path_dict:
        result_path = data_path_dict[data_paths]['path']
        infer_speed_path = data_path_dict[data_paths]['infer_speed']
        results.append(get_single_ray.remote(
            result_path, infer_speed_path
        ))
    result = ray.get(results)
    print('finished ray...')
    result = [r for r in result if r is not None]
    return result


def get_s3_paths(path_prefix: str, contains=None, suffix=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix=suffix, contains=contains)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    return paths_full


def get_single(path, path_infer_speed):
    dataset_directory = path.rsplit('/', 2)[0]
    task_directory = dataset_directory.rsplit('/', 2)[1]
    print(task_directory)

    scores = load_pd.load(path)
    try:
        infer_speed_df = load_pd.load(path_infer_speed)
    except Exception:
        print(f'MISS2: {path_infer_speed}')
        return None
    else:
        print(f'YES: {path_infer_speed}')
        infer_speed_best = infer_speed_df[infer_speed_df['model'] == 'best']
        batch_sizes = list(infer_speed_best['batch_size'].unique())
        for batch_size in batch_sizes:
            scores[f'pred_time_test_with_transform_batch_size_{batch_size}'] = \
            infer_speed_best[infer_speed_best['batch_size'] == batch_size].iloc[0]['pred_time_test_with_transform']
        return scores


def aggregate_infer_speed(path_prefix: str, contains=None):
    paths_full = get_s3_paths(path_prefix, contains=contains, suffix='/scores/results.csv')
    paths_infer_speed = get_s3_paths(path_prefix, suffix='/infer_speed.csv')
    data_path_dict = {}
    for path in paths_full:
        dataset_directory = path.rsplit('/', 2)[0]
        data_path_dict[dataset_directory] = {}
        data_path_dict[dataset_directory]['path'] = path
    for path in paths_infer_speed:
        for k in data_path_dict:
            if path.startswith(k):
                if 'infer_speed' in data_path_dict[k]:
                    raise AssertionError('Multiple infer_speed files found!')
                data_path_dict[k]['infer_speed'] = path
                break
    paths = [p for p in data_path_dict.keys()]
    for path in paths:
        if 'infer_speed' not in data_path_dict[path]:
            print(f'MISSING: {path}')
            data_path_dict.pop(path, None)

    df_full = select_ray(data_path_dict=data_path_dict)

    df_full = pd.concat(df_full, ignore_index=True)
    print(df_full)
    return df_full


def aggregate_leaderboards_from_params(s3_bucket, s3_prefix, version_name, constraint):
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'
    aggregated_results_name = f'results_automlbenchmark_{constraint}_{version_name}_infer_speed.csv'

    df = aggregate_infer_speed(path_prefix=f's3://{s3_bucket}/{result_path}', contains=contains)

    save_path = f's3://{s3_bucket}/aggregated/{result_path}{aggregated_results_name}'
    save_pd.save(path=save_path, df=df)
    print(f'Saved to {save_path}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default='1h8c', nargs='?')
    parser.set_defaults(keep_params=True)
    parser.set_defaults(version_name="2023_02_20_bool_test")  # FIXME: Remove
    args = parser.parse_args()


    aggregate_leaderboards_from_params(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint=args.constraint,
    )
