import argparse

import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.loaders.load_s3 import list_bucket_prefix_suffix_s3, list_bucket_prefix_suffix_contains_s3
from autogluon.common.utils import s3_utils
from autogluon.common.savers import save_pd


def get_s3_paths(path_prefix: str, contains=None, suffix=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix=suffix, contains=contains)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    return paths_full


def aggregate_leaderboards(path_prefix: str, contains=None, keep_params=True):
    paths_full = get_s3_paths(path_prefix, contains=contains, suffix='/scores/results.csv')

    df_full = []
    num_paths = len(paths_full)
    columns_to_keep = ['id', 'task', 'framework', 'constraint', 'fold', 'type', 'metric', 'mode', 'version', 'params', 'app_version', 'utc', 'seed']

    if not keep_params:
        columns_to_keep.remove('params')
    print(keep_params)
    print(columns_to_keep)

    for i, path in enumerate(paths_full):
        print(f'{i+1}/{num_paths} | {path}')
        dataset_directory = path.rsplit('/', 2)[0]
        path_leaderboard = get_s3_paths(dataset_directory, suffix='/leaderboard.csv')
        if len(path_leaderboard) != 1:
            continue
        else:
            path_leaderboard = path_leaderboard[0]
        print(path_leaderboard)
        scores = load_pd.load(path)
        try:
            leaderboard = load_pd.load(path_leaderboard)
            leaderboard = leaderboard.drop(columns=['features'], errors='ignore')
        except Exception:
            continue
        else:
            # scores = scores[scores['fold'] == 0]
            print(scores)
            scores = scores[columns_to_keep]
            scores = scores.rename(columns={'framework': 'framework_parent'})

            # best_compressed = leaderboard[leaderboard['model'].str.contains('_FULL')]
            # best_distilled = leaderboard[leaderboard['model'].str.contains('_d1')].sort_values('score_val', ascending=False).head(1)
            best_weighted = leaderboard[leaderboard['model'].str.contains('WeightedEnsemble_')].sort_values('score_val', ascending=False).head(1)
            best_nonweighted = leaderboard[~leaderboard['model'].str.contains('WeightedEnsemble_')].sort_values('score_val', ascending=False).head(1)

            # best_compressed['model'] = 'autogluon_compressed'
            # best_distilled['model'] = 'autogluon_distilled'
            best_weighted['model'] = 'autogluon_ensemble'
            best_nonweighted['model'] = 'autogluon_single'
            # print(best_compressed)
            # print(best_distilled)
            # print(best_weighted)

            combined = pd.concat([
                leaderboard,
                best_weighted,
                # best_compressed,
                # best_distilled,
                best_nonweighted,
            ], ignore_index=True)
            # combined = combined.sort_values('score_test', ascending=False).reset_index(drop=True)
            combined['id'] = scores['id'][0]
            print(combined)

            combined_full = pd.merge(combined, scores, on='id', how='left')
            # print(combined_full)
            df_full.append(combined_full)

    df_full = pd.concat(df_full, ignore_index=True)
    df_full['framework'] = df_full['model']
    # df_full['result'] = df_full['score_test']
    df_full['duration'] = df_full['fit_time']
    # df_full['predict_duration'] = df_full['pred_time_test']
    print(df_full)
    return df_full


def aggregate_leaderboards_from_params(s3_bucket, s3_prefix, version_name, constraint, keep_params=True):
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'
    aggregated_results_name = f'results_ag_leaderboard_{constraint}_{version_name}.csv'

    df = aggregate_leaderboards(path_prefix=f's3://{s3_bucket}/{result_path}', contains=contains, keep_params=keep_params)

    save_pd.save(path=f's3://{s3_bucket}/aggregated/{result_path}{aggregated_results_name}', df=df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default='1h8c', nargs='?')
    parser.add_argument('--keep_params', action='store_true')
    parser.add_argument('--no-keep_params', dest='keep_params', action='store_false')
    parser.set_defaults(keep_params=True)
    args = parser.parse_args()

    aggregate_leaderboards_from_params(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint=args.constraint,
        keep_params=args.keep_params,
    )
