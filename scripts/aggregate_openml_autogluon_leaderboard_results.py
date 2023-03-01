import argparse
import time
from typing import Dict, List

import pandas as pd
import ray

from autogluon.common.loaders import load_pd, load_s3
from autogluon.common.utils import s3_utils
from autogluon.common.savers import save_pd

from autogluon_benchmark import OutputContext


@ray.remote
def get_single_leaderboard_ray(output_context, columns_to_keep, with_infer_speed, i, num_paths):
    return get_single_leaderboard(output_context, columns_to_keep, with_infer_speed, i, num_paths)


def select_ray(paths_full, columns_to_keep, with_infer_speed):
    num_paths = len(paths_full)

    if len(paths_full) == 0:
        raise AssertionError('Empty paths_full!')

    paths_dict = {path: OutputContext.from_result_path(path=path) for path in paths_full}
    print('starting ray...')
    # Create and execute all tasks in parallel
    if not ray.is_initialized():
        ray.init()
    results = []
    for i, path in enumerate(paths_dict):
        output_context = paths_dict[path]
        results.append(get_single_leaderboard_ray.remote(
            output_context, columns_to_keep, with_infer_speed, i, num_paths
        ))
    result = ray.get(results)
    print('finished ray...')
    result = [r for r in result if r is not None]
    return result


def get_s3_paths(path_prefix: str, contains=None, suffix=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = load_s3.list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix=suffix, contains=contains)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    return paths_full


def get_single_leaderboard(output_context: OutputContext, columns_to_keep, with_infer_speed, i, num_paths):
    print(f'{i + 1}/{num_paths} | {output_context.path}')
    dataset_directory = output_context.path
    scores = load_pd.load(output_context.path_result)
    try:
        leaderboard = load_pd.load(output_context.path_leaderboard)
        leaderboard = leaderboard.drop(columns=['features'], errors='ignore')
        if with_infer_speed:
            leaderboard = merge_with_infer_speed(leaderboard=leaderboard, dataset_directory=dataset_directory, path_infer_speed=output_context.path_infer_speed)
    except Exception:
        return None
    else:
        # scores = scores[scores['fold'] == 0]
        # print(scores)
        scores = scores[columns_to_keep]
        scores = scores.rename(columns={'framework': 'framework_parent'})

        # best_compressed = leaderboard[leaderboard['model'].str.contains('_FULL')]
        # best_distilled = leaderboard[leaderboard['model'].str.contains('_d1')].sort_values('score_val', ascending=False).head(1)
        best_weighted = leaderboard[leaderboard['model'].str.contains('WeightedEnsemble_')].sort_values('score_val',
                                                                                                        ascending=False).head(
            1)
        best_nonweighted = leaderboard[~leaderboard['model'].str.contains('WeightedEnsemble_')].sort_values('score_val',
                                                                                                            ascending=False).head(
            1)

        # best_compressed['model'] = 'autogluon_compressed'
        # best_distilled['model'] = 'autogluon_distilled'
        # FIXME: Doesn't work for refit_full!!! score_val is NaN!
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
        # print(combined)

        combined_full = pd.merge(combined, scores, on='id', how='left')
        # print(combined_full)
        print(f'{i} done')
        return combined_full


def aggregate_leaderboards(path_prefix: str, contains=None, keep_params=True, with_infer_speed=False):
    paths_full = get_s3_paths(path_prefix, contains=contains, suffix='scores/results.csv')
    columns_to_keep = ['id', 'task', 'framework', 'constraint', 'fold', 'type', 'metric', 'mode', 'version', 'params', 'app_version', 'utc', 'seed']

    if not keep_params:
        columns_to_keep.remove('params')
    print(f'columns_to_keep = {columns_to_keep}')

    df_full = select_ray(
        paths_full=paths_full,
        columns_to_keep=columns_to_keep,
        with_infer_speed=with_infer_speed
    )

    df_full = pd.concat(df_full, ignore_index=True)
    df_full['framework'] = df_full['model']
    # df_full['result'] = df_full['score_test']
    df_full['duration'] = df_full['fit_time']
    # df_full['predict_duration'] = df_full['pred_time_test']
    print(df_full)
    return df_full


def merge_with_infer_speed(leaderboard, dataset_directory, path_infer_speed=None) -> pd.DataFrame:
    if path_infer_speed is None:
        path_infer_speed = get_s3_paths(dataset_directory, suffix='/infer_speed.csv')
        if len(path_infer_speed) != 1:
            print(f'MISS: {dataset_directory}')
            raise AssertionError(f'Invalid infer_speed paths in {dataset_directory}: {path_infer_speed}')
        path_infer_speed = path_infer_speed[0]

    infer_speed_df = load_pd.load(path_infer_speed)
    infer_speed_df = infer_speed_df[['model', 'batch_size', 'pred_time_test_with_transform']]
    infer_speed_m_df = infer_speed_df.set_index(['model', 'batch_size'], drop=True)
    a = infer_speed_m_df.to_dict()
    b = a['pred_time_test_with_transform']
    c = dict()
    for key_pair, pred_time_test_with_transform in b.items():
        m = key_pair[0]
        bs = key_pair[1]
        col_name = f'pred_time_test_with_transform_{bs}'
        if col_name not in c:
            c[col_name] = {}
        c[col_name][m] = pred_time_test_with_transform
    c_df = pd.DataFrame(c).rename_axis('model').reset_index(drop=False)
    leaderboard = pd.merge(leaderboard, c_df, on='model')
    return leaderboard


def aggregate_leaderboards_from_params(s3_bucket, s3_prefix, version_name, constraint, keep_params=True, with_infer_speed=False):
    ts = time.time()
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'
    aggregated_results_name = f'results_ag_leaderboard_{constraint}_{version_name}.csv'

    df = aggregate_leaderboards(path_prefix=f's3://{s3_bucket}/{result_path}', contains=contains, keep_params=keep_params, with_infer_speed=with_infer_speed)

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
    parser.add_argument('--with_infer_speed', action='store_true')
    parser.add_argument('--no-with_infer_speed', dest='with_infer_speed', action='store_false')
    parser.set_defaults(keep_params=True)
    parser.set_defaults(with_infer_speed=False)

    # parser.set_defaults(version_name="2022_11_14_v06")
    parser.set_defaults(version_name="2023_02_27_zs")
    args = parser.parse_args()

    aggregate_leaderboards_from_params(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint='24h64c',
        keep_params=False,
        with_infer_speed=args.with_infer_speed,
    )
