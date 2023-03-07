import argparse
import copy
import time

from autogluon.common.savers import save_pd

from autogluon_benchmark import OutputSuiteContext
from autogluon_benchmark.benchmark_context._output_suite_context import DEFAULT_COLUMNS_TO_KEEP


def aggregate_leaderboards(path_prefix: str, contains=None, keep_params=True, include_infer_speed=False, mode='seq'):
    columns_to_keep: list = copy.deepcopy(DEFAULT_COLUMNS_TO_KEEP)

    if not keep_params:
        columns_to_keep.remove('params')
    print(f'columns_to_keep = {columns_to_keep}')

    output_suite_context = OutputSuiteContext(path=path_prefix,
                                              contains=contains,
                                              columns_to_keep=columns_to_keep,
                                              include_infer_speed=include_infer_speed,
                                              mode=mode)
    output_suite_context.filter_failures()

    # output_suite_context.get_benchmark_failures()

    df_full = output_suite_context.aggregate_leaderboards()
    print(df_full)
    return df_full


def aggregate_leaderboards_from_params(s3_bucket,
                                       s3_prefix,
                                       version_name,
                                       constraint,
                                       keep_params=True,
                                       include_infer_speed=False,
                                       mode='seq'):
    ts = time.time()
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'
    aggregated_results_name = f'results_ag_leaderboard_{constraint}_{version_name}.csv'

    df = aggregate_leaderboards(path_prefix=f's3://{s3_bucket}/{result_path}',
                                contains=contains,
                                keep_params=keep_params,
                                include_infer_speed=include_infer_speed,
                                mode=mode)

    save_path = f's3://{s3_bucket}/aggregated/{result_path}{aggregated_results_name}'

    print(f'Saving output to "{save_path}"')

    save_pd.save(path=save_path, df=df)

    print(f'Success! Saved output to "{save_path}"')
    te = time.time()
    print(f'Total Time Taken: {round(te-ts, 2)}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default='1h8c', nargs='?')
    parser.add_argument('--keep_params', action='store_true')
    parser.add_argument('--no-keep_params', dest='keep_params', action='store_false')
    parser.add_argument('--include_infer_speed', action='store_true')
    parser.add_argument('--no-include_infer_speed', dest='include_infer_speed', action='store_false')
    parser.add_argument('--mode', type=str, help='Whether to aggregate via "seq" or via "ray"', default='ray',
                        nargs='?')
    parser.set_defaults(keep_params=True)
    parser.set_defaults(include_infer_speed=False)

    # parser.set_defaults(version_name="2022_11_14_v06")
    parser.set_defaults(version_name="2023_02_27_zs")
    args = parser.parse_args()

    aggregate_leaderboards_from_params(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint='24h64c',
        keep_params=False,
        include_infer_speed=args.include_infer_speed,
        mode=args.mode,
    )
