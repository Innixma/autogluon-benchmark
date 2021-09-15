import json
import argparse
from autogluon.core.utils import s3_utils
from autogluon.core.utils.loaders import load_pd
from autogluon.core.utils.loaders.load_s3 import list_bucket_prefix_suffix_contains_s3
from autogluon.core.utils.savers import save_pd


def aggregate(path_prefix: str, contains=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    # objects = list_bucket_prefix_suffix_s3(bucket=bucket, prefix=prefix, suffix='scores/results.csv')
    print(f'{bucket} | {prefix} | {contains}')
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix='scores/results.csv',
                                                    contains=contains)
    print(objects)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    print(paths_full)
    df = load_pd.load(paths_full)
    print(df)
    return df


def aggregate_from_params(s3_bucket, s3_prefix, version_name, suffix, contains, results_prefix='results/',
                          aggregated_prefix='aggregated/', save_path_str_replace_dict=None):
    result_path = s3_prefix + version_name + '/'
    aggregated_results_name = 'results_automlbenchmark' + suffix + '_' + version_name + '.csv'

    df = aggregate(path_prefix='s3://' + s3_bucket + '/' + results_prefix + result_path, contains=contains)

    save_path = 's3://' + s3_bucket + '/' + aggregated_prefix + result_path + aggregated_results_name
    if save_path_str_replace_dict is not None:
        for substring, replacement in save_path_str_replace_dict.items():
            save_path = save_path.replace(substring, replacement)
    save_pd.save(path=save_path, df=df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='', nargs='?')
    parser.add_argument('version_name', type=str, help='Root folder name in EC2 of results',
                        default='ec2/2021_06_08_holdout', nargs='?')
    parser.add_argument('suffix', type=str, help='Suffix in aggregated results csv name',
                        default='_1h8c', nargs='?')
    parser.add_argument('contains', type=str, help='Results must contain this term',
                        default='.1h8c.', nargs='?')
    parser.add_argument('results_prefix', type=str, help='Name of folder one level above where results csv are saved',
                        default='', nargs='?')
    parser.add_argument('save_path_str_replace_dict', type=str, help='Dictionary in string form of string values to replace(dict key) with another string value(dict value)',
                        default="{\'_ec2/\': \'_\'}", nargs='?')

    args = parser.parse_args()

    # Str to Json dict requires properties to be in double quotes
    save_path_str_replace_str = args.save_path_str_replace_dict.replace('\'', '\"')
    save_path_str_replace_dict = json.loads(save_path_str_replace_str)

    aggregate_from_params(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        suffix=args.suffix,
        contains=args.contains,
        results_prefix=args.results_prefix,
        save_path_str_replace_dict=save_path_str_replace_dict,
    )

