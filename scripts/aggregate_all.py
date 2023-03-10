import argparse

from autogluon.common.savers import save_pd, save_pkl
from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from autogluon_benchmark import OutputSuiteContext
from autogluon_benchmark.aggregate.zeroshot_metadata import load_zeroshot_metadata


def aggregate_all(path_prefix,
                  version_name=None,
                  constraint=None,
                  include_infer_speed=False,
                  aggregate_zeroshot=False,
                  aggregate_leaderboard=False,
                  keep_params=False,
                  invalid_datasets=None,
                  folds=None,
                  max_size_mb=100,
                  mode='ray'):
    s3_bucket, s3_prefix = s3_path_to_bucket_prefix(s3_path=path_prefix)
    output_path = f's3://{s3_bucket}/aggregated/{s3_prefix}'
    constraint_str = f'_{constraint}' if constraint is not None else ''
    version_name_str = f'_{version_name}' if version_name is not None else ''

    if constraint is not None:
        contains = f'.{constraint}.'
    else:
        contains = None

    output_suite_context = OutputSuiteContext(
        path=path_prefix,
        contains=contains,
        include_infer_speed=include_infer_speed,
        keep_params=keep_params,
        mode=mode,
    )
    results_df = output_suite_context.aggregate_results()
    output_suite_context.filter_failures()
    if aggregate_leaderboard:
        leaderboards_df = output_suite_context.aggregate_leaderboards()
    else:
        leaderboards_df = None

    if aggregate_zeroshot:
        aggregated_pred_proba, aggregated_ground_truth = load_zeroshot_metadata(
            output_suite_context=output_suite_context,
            invalid_datasets=invalid_datasets,
            folds=folds,
            max_size_mb=max_size_mb,
        )
    else:
        aggregated_pred_proba = None
        aggregated_ground_truth = None

    aggregated_results_name = f'results{constraint_str}{version_name_str}.csv'
    aggregated_results_path = f'{output_path}{aggregated_results_name}'
    save_pd.save(path=aggregated_results_path, df=results_df)
    print(f'Saved to {aggregated_results_path}!')
    if aggregate_leaderboard:
        aggregated_leaderboard_name = f'leaderboard{constraint_str}{version_name_str}.csv'
        aggregated_leaderboard_path = f'{output_path}{aggregated_leaderboard_name}'
        print(f'Saving output to "{aggregated_leaderboard_path}"')
        save_pd.save(path=aggregated_leaderboard_path, df=leaderboards_df)
        print(f'Success! Saved output to "{aggregated_leaderboard_path}"')

    if aggregate_zeroshot:
        max_mb_str = f'_{int(max_size_mb)}_mb' if max_size_mb is not None else ''

        aggregated_pred_proba_path = f'{output_path}zeroshot_pred_proba{max_mb_str}.pkl'
        aggregated_ground_truth_path = f'{output_path}zeroshot_gt{max_mb_str}.pkl'
        print(f'Saving pred_proba output to {aggregated_pred_proba_path}')
        print(f'Saving ground_truth output to {aggregated_ground_truth_path}')

        save_pkl.save(path=aggregated_pred_proba_path, object=aggregated_pred_proba)
        save_pkl.save(path=aggregated_ground_truth_path, object=aggregated_ground_truth)

    pass


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
    parser.set_defaults(keep_params=False)
    parser.set_defaults(include_infer_speed=False)
    parser.set_defaults(version_name="2023_02_27_zs")  # FIXME: Remove
    parser.set_defaults(constraint="24h64c")  # FIXME: Remove
    args = parser.parse_args()

    path_prefix = f's3://{args.s3_bucket}/{args.s3_prefix}{args.version_name}/'

    aggregate_all(
        path_prefix=path_prefix,
        # version_name=args.version_name,
        # constraint=args.constraint,
        include_infer_speed=args.include_infer_speed,
        keep_params=args.keep_params,
        aggregate_leaderboard=True,
        aggregate_zeroshot=True,
        mode=args.mode,
    )
