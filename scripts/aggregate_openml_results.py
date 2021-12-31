import argparse

from autogluon.common.utils import s3_utils
from autogluon.common.loaders import load_pd
from autogluon.common.loaders.load_s3 import list_bucket_prefix_suffix_contains_s3
from autogluon.common.savers import save_pd


def aggregate(path_prefix: str, contains=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    # objects = list_bucket_prefix_suffix_s3(bucket=bucket, prefix=prefix, suffix='scores/results.csv')
    print(f'{bucket} | {prefix} | {contains}')
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix='scores/results.csv',
                                                    contains=contains)
    print(f'found {len(objects)} objects')
    print('Printing first 3 objects:')
    print(objects[:3])
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    print(paths_full[:3])
    df = load_pd.load(paths_full)
    print(df)
    return df


def aggregate_from_params(s3_bucket, s3_prefix, version_name, constraint, aggregated_prefix='aggregated/'):
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'
    aggregated_results_name = f'results_automlbenchmark_{constraint}_{version_name}.csv'

    df = aggregate(path_prefix=f's3://{s3_bucket}/{result_path}', contains=contains)

    save_path = f's3://{s3_bucket}/{aggregated_prefix}{result_path}{aggregated_results_name}'

    print(f'Saving to: {save_path}')
    save_pd.save(path=save_path, df=df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default='1h8c', nargs='?')

    args = parser.parse_args()

    aggregate_from_params(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint=args.constraint,
    )
