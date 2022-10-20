import argparse

import pandas as pd

from autogluon.common.loaders import load_pd, load_pkl
from autogluon.common.loaders.load_s3 import list_bucket_prefix_suffix_s3, list_bucket_prefix_suffix_contains_s3
from autogluon.common.utils import s3_utils
from autogluon.common.savers import save_pd


def get_s3_paths(path_prefix: str, contains=None, suffix=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix=suffix, contains=contains)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    return paths_full


def aggregate_zeroshot_metadata(path_prefix: str, contains=None):
    paths_full = get_s3_paths(path_prefix, contains=contains, suffix='/scores/results.csv')

    df_full = []
    num_paths = len(paths_full)

    for i, path in enumerate(paths_full):
        print(f'{i+1}/{num_paths} | {path}')
        dataset_directory = path.rsplit('/', 2)[0]
        path_metadata = get_s3_paths(dataset_directory, suffix='/zeroshot_metadata.pkl')
        print(path_metadata)
        if len(path_metadata) != 1:
            continue
        else:
            path_metadata = path_metadata[0]
        print(path_metadata)
        zeroshot_metadata = None
        try:
            zeroshot_metadata = load_pkl.load(path)
        except Exception:
            continue
        else:
            print(zeroshot_metadata)
    return 'todo'


def aggregate_leaderboards_from_params(s3_bucket, s3_prefix, version_name, constraint):
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'
    # aggregated_results_name = f'results_ag_zeroshot_{constraint}_{version_name}.csv'

    df = aggregate_zeroshot_metadata(path_prefix=f's3://{s3_bucket}/{result_path}', contains=contains)

    # save_pd.save(path=f's3://{s3_bucket}/aggregated/{result_path}{aggregated_results_name}', df=df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default='1h8c', nargs='?')
    parser.add_argument('--keep_params', action='store_true')
    args = parser.parse_args()

    aggregate_leaderboards_from_params(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint=args.constraint,
    )
