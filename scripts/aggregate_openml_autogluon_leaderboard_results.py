import pandas as pd

from autogluon.core.utils.loaders import load_pd
from autogluon.core.utils.loaders.load_s3 import list_bucket_prefix_suffix_s3, list_bucket_prefix_suffix_contains_s3
from autogluon.core.utils import s3_utils
from autogluon.core.utils.savers import save_pd


def get_s3_paths(path_prefix: str, contains=None, suffix=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix=suffix, contains=contains)
    print(objects)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    print(paths_full)
    return paths_full


def aggregate_leaderboards(path_prefix: str, contains=None):
    paths_full = get_s3_paths(path_prefix, contains=contains, suffix='/scores/results.csv')

    df_full = []
    for path in paths_full:
        print(path)
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
            scores = scores[['id', 'task', 'framework', 'constraint', 'fold', 'type', 'metric', 'mode', 'version', 'params', 'app_version', 'utc', 'seed']]
            scores = scores.rename(columns={'framework': 'framework_parent'})

            best_compressed = leaderboard[leaderboard['model'].str.contains('_FULL')]
            best_distilled = leaderboard[leaderboard['model'].str.contains('_d1')].sort_values('score_val', ascending=False).head(1)
            best_weighted = leaderboard[leaderboard['model'].str.contains('WeightedEnsemble_')].sort_values('score_val', ascending=False).head(1)
            best_nonweighted = leaderboard[~leaderboard['model'].str.contains('WeightedEnsemble_')].sort_values('score_val', ascending=False).head(1)

            best_compressed['model'] = 'autogluon_compressed'
            best_distilled['model'] = 'autogluon_distilled'
            best_weighted['model'] = 'autogluon_ensemble'
            best_nonweighted['model'] = 'autogluon_single'
            # print(best_compressed)
            # print(best_distilled)
            # print(best_weighted)

            combined = pd.concat([
                leaderboard,
                best_weighted,
                best_compressed,
                best_distilled,
                best_nonweighted,
            ], ignore_index=True)
            # combined = combined.sort_values('score_test', ascending=False).reset_index(drop=True)
            combined['id'] = scores['id'][0]
            print(combined)

            combined_full = pd.merge(combined, scores, on='id', how='left')
            print(combined_full)
            df_full.append(combined_full)

    df_full = pd.concat(df_full, ignore_index=True)
    df_full['framework'] = df_full['model']
    # df_full['result'] = df_full['score_test']
    df_full['duration'] = df_full['fit_time']
    # df_full['predict_duration'] = df_full['pred_time_test']
    print(df_full)
    return df_full


def aggregate_leaderboards_from_params(s3_bucket, s3_prefix, version_name, suffix, contains):
    result_path = s3_prefix + version_name + '/'
    aggregated_results_name = 'results_ag_leaderboard' + suffix + '_' + version_name + '.csv'

    df = aggregate_leaderboards(path_prefix='s3://' + s3_bucket + '/' + result_path, contains=contains)

    save_pd.save(path='s3://' + s3_bucket + '/aggregated/' + result_path + aggregated_results_name, df=df)


if __name__ == '__main__':
    aggregate_leaderboards_from_params(
        s3_bucket='automl-benchmark-ag',
        s3_prefix='ec2/',
        version_name='2021_05_22_infopt',
        suffix='_1h8c',
        contains='.1h8c.',
    )
