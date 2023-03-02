import argparse

from autogluon.common.savers import save_pd

from autogluon_benchmark import OutputSuiteContext


def aggregate_results(s3_bucket,
                      s3_prefix,
                      version_name,
                      constraint,
                      include_infer_speed=False,
                      mode='ray'):
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'
    path_prefix = f's3://{s3_bucket}/{result_path}'

    aggregated_results_name = f'results_automlbenchmark_{constraint}_{version_name}.csv'

    output_suite_context = OutputSuiteContext(
        path=path_prefix,
        contains=contains,
        include_infer_speed=include_infer_speed,
        mode=mode,
    )
    results_df = output_suite_context.aggregate_results()
    print(results_df)

    save_path = f's3://{s3_bucket}/aggregated/{result_path}{aggregated_results_name}'
    save_pd.save(path=save_path, df=results_df)
    print(f'Saved to {save_path}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default='1h8c', nargs='?')
    parser.add_argument('--include_infer_speed', action='store_true')
    parser.add_argument('--no-include_infer_speed', dest='include_infer_speed', action='store_false')
    parser.add_argument('--mode', type=str, help='Whether to aggregate via "seq" or via "ray"', default='ray',
                        nargs='?')
    parser.set_defaults(keep_params=True)
    parser.set_defaults(include_infer_speed=False)
    parser.set_defaults(version_name="2023_02_20_bool_test")  # FIXME: Remove
    args = parser.parse_args()

    aggregate_results(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint=args.constraint,
        include_infer_speed=args.include_infer_speed,
        mode=args.mode,
    )
