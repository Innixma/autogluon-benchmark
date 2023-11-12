import copy
from typing import Optional, Set

import argparse

import pandas as pd
from tqdm import tqdm

from autogluon.common.savers import save_pd, save_pkl, save_str
from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from autogluon.bench.eval.benchmark_context.output_suite_context import OutputSuiteContext
from autogluon.bench.eval.evaluation.metadata.metadata_loader import load_task_metadata
from autogluon.bench.eval.scripts.run_generate_clean_openml import clean_and_save_results

from autogluon_benchmark.aggregate.zeroshot_metadata import load_zeroshot_metadata


def get_dataset_size_mb(
        leaderboards_df: pd.DataFrame,
        results_valid_df: pd.DataFrame,
        output_suite_context: OutputSuiteContext) -> pd.DataFrame:
    """
    Returns the size in megabytes of all zeroshot metadata artifacts for a given dataset (across all folds)

    Parameters
    ----------
    leaderboards_df
    results_valid_df
    output_suite_context

    Returns
    -------
    A pandas DataFrame with two columns: ["dataset", "size_mb"], sorted by descending "size_mb".

    """
    leaderboards_df_tmp = copy.deepcopy(leaderboards_df)
    unique_tasks_dict = leaderboards_df_tmp[['task', 'fold']].drop_duplicates().groupby('task')['fold'].apply(set).to_dict()

    dataset_size_mb_dict = {}
    for task, task_folds in tqdm(unique_tasks_dict.items()):
        total_size = 0
        for fold in task_folds:
            result_task_fold_indices = (results_valid_df["task"] == task) & (results_valid_df["fold"] == fold)
            output_suite_context_task_fold = output_suite_context.filter(filter_lst=list(result_task_fold_indices), inplace=False)
            zeroshot_metadata_size_bytes = output_suite_context_task_fold.get_zeroshot_metadata_size_bytes(allow_exception=True)
            for b in zeroshot_metadata_size_bytes:
                if b is not None:
                    total_size += b
        total_size_mb = total_size / 1e6
        dataset_size_mb_dict[task] = total_size_mb

    sorted_tuple_list = sorted(dataset_size_mb_dict.items(), key=lambda x: x[1], reverse=True)
    for d, mb in sorted_tuple_list:
        print(f"{mb:.3f} MB\t{d}")
    dataset_size_mb_df = pd.DataFrame(sorted_tuple_list, columns=["dataset", "size_mb"])
    return dataset_size_mb_df


def save_zs_metadata_per_task(
        results_valid_df: pd.DataFrame,
        output_suite_context: OutputSuiteContext,
        unique_tasks_dict: dict = None,
        banned_tasks: list = None,
):

    if unique_tasks_dict is None:
        unique_tasks_dict = results_valid_df[["task", "fold"]].groupby("task")["fold"].agg(set).to_dict()

    for task, task_folds in unique_tasks_dict.items():
        if banned_tasks:
            if task in banned_tasks:
                continue
        for fold in task_folds:
            output_path_task_fold = output_path + f'zeroshot_metadata/{task}/{fold}/'
            print(f'{task}\t{fold}')
            result_task_fold_indices = (results_valid_df["task"] == task) & (results_valid_df["fold"] == fold)
            result_task_fold_df = results_valid_df[result_task_fold_indices]

            output_suite_context_task_fold = output_suite_context.filter(filter_lst=list(result_task_fold_indices), inplace=False)

            # Get overall size
            # zeroshot_metadata_size_bytes = output_suite_context_task_fold.get_zeroshot_metadata_size_bytes(allow_exception=True)
            # total_size = 0
            # for b in zeroshot_metadata_size_bytes:
            #     if b is not None:
            #         total_size += b
            #
            # total_size_mb = total_size / 1e6
            # print(f"\t{total_size_mb:.3f} MB")

            aggregated_pred_proba, aggregated_ground_truth = load_zeroshot_metadata(
                output_suite_context=output_suite_context_task_fold,
                max_size_mb=None,
            )

            aggregated_pred_proba_path = f'{output_path_task_fold}zeroshot_pred_proba.pkl'
            aggregated_ground_truth_path = f'{output_path_task_fold}zeroshot_gt.pkl'
            print(f'Saving output to {output_path_task_fold}')

            save_pkl.save(path=aggregated_pred_proba_path, object=aggregated_pred_proba)
            save_pkl.save(path=aggregated_ground_truth_path, object=aggregated_ground_truth)


def aggregate_all(path_prefix,
                  version_name=None,
                  use_version_name_str=False,
                  constraint=None,
                  contains=None,
                  allowed_tids: Optional[Set[int]] = None,
                  include_infer_speed=False,
                  aggregate_zeroshot=False,
                  aggregate_leaderboard=False,
                  aggregate_model_failures=False,
                  aggregate_logs=False,
                  output_path=None,
                  keep_params=False,
                  invalid_datasets=None,
                  folds=None,
                  max_size_mb=100,
                  mode='ray'):
    s3_bucket, s3_prefix = s3_path_to_bucket_prefix(s3_path=path_prefix)
    if output_path is None:
        output_path = f's3://{s3_bucket}/aggregated/{s3_prefix}'
    constraint_str = f'_{constraint}' if constraint is not None else ''
    if use_version_name_str:
        version_name_str = f'_{version_name}' if version_name is not None else ''
    else:
        version_name_str = ''

    if constraint is not None and contains is None:
        contains = f'.{constraint}.'

    print(f'Initializing OutputSuiteContext from path="{path_prefix}"')
    output_suite_context = OutputSuiteContext(
        path=path_prefix,
        contains=contains,
        allowed_tids=allowed_tids,
        include_infer_speed=include_infer_speed,
        keep_params=keep_params,
        mode=mode,
    )
    print(f'Fetching results from {output_suite_context.num_contexts} OutputContexts')

    sample_size = None
    if sample_size is not None and sample_size < output_suite_context.num_contexts:
        filter_lst = [i < sample_size for i in range(output_suite_context.num_contexts)]
        output_suite_context.filter(filter_lst=filter_lst)
        print(f'Filtered to {sample_size} contexts (sampling)')

    results_df_list = output_suite_context.load_results()
    len_results = len(results_df_list)
    if folds is not None:
        for i in range(len_results):
            result = results_df_list[i]
            if result is not None:
                if result.iloc[0]['fold'] not in folds:
                    results_df_list[i] = None

    output_suite_context.filter(filter_lst=[result is not None for result in results_df_list])
    print(f'Filtered to {output_suite_context.num_contexts} results '
          f'(Filtered results may have TIDs missing from `allowed_tids`)')

    if aggregate_logs:
        aggregated_logs_name = f'logs{constraint_str}{version_name_str}.txt'
        aggregated_logs_path = f'{output_path}{aggregated_logs_name}'
        print(f'Saving logs to "{aggregated_logs_path}"')
        logs = output_suite_context.aggregate_logs()
        save_str.save(path=aggregated_logs_path, data=logs)
        print(f'Success! Saved logs to "{aggregated_logs_path}"')

    if aggregate_model_failures:
        model_failures_df = output_suite_context.aggregate_model_failures()
        if model_failures_df is not None:
            aggregated_model_failures_name = f'model_failures{constraint_str}{version_name_str}.csv'
            aggregated_model_failures_path = f'{output_path}{aggregated_model_failures_name}'
            print(f'Saving output to "{aggregated_model_failures_path}"')
            save_pd.save(path=aggregated_model_failures_path, df=model_failures_df)
            print(f'Success! Saved output to "{aggregated_model_failures_path}"')

    results_df = output_suite_context.aggregate_results(results_list=results_df_list)
    output_suite_context.filter_failures()

    # Get the results df for only valid (successful) jobs
    results_valid_df = output_suite_context.aggregate_results()  # TODO: Can speedup by computing from `results_df`

    # FIXME: Drop duplicates before aggregating leaderboard / zeroshot

    if aggregate_leaderboard:
        leaderboards_list = output_suite_context.load_leaderboards()
        leaderboards_df = output_suite_context.aggregate_leaderboards(leaderboards_list)
    else:
        leaderboards_df = None

    unique_tasks_dict = results_valid_df[["task", "fold"]].groupby("task")["fold"].agg(set).to_dict()

    banned_tasks = [
        "dionis",
        "KDDCup99",
        "Kuzushiji-49",
        "Airlines_DepDelay_10M",
        "pokerhand",
        "sf-police-incidents",
        "helena",
        "covertype",
        "Devnagari-Script",
        "Higgs",
        "walking-activity",
        "spoken-arabic-digit",
        "GTSRB-HOG01",
        "GTSRB-HOG02",
        "GTSRB-HOG03",
        "GTSRB-HueHist",
    ]

    if leaderboards_df is not None:
        # Dedupe leaderboard, keeping latest result in UTC
        leaderboards_df_dedupe = leaderboards_df.sort_values(
            by=["utc"], ascending=False
        ).drop_duplicates(
            subset=["model", "task", "fold"], keep="first"
        ).sort_index()

        leaderboards_df_only_zs = leaderboards_df_dedupe[leaderboards_df_dedupe["framework_parent"].str.contains("ZS_BAG_")]
        leaderboards_df_only_zs = leaderboards_df_only_zs[leaderboards_df_only_zs["model"] != "autogluon_single"]

    aggregated_results_name = f'results{constraint_str}{version_name_str}.csv'
    aggregated_results_path = f'{output_path}{aggregated_results_name}'
    aggregated_results_valid_name = f'results_valid{constraint_str}{version_name_str}.csv'
    aggregated_results_valid_path = f'{output_path}{aggregated_results_valid_name}'
    save_pd.save(path=aggregated_results_path, df=results_df)
    save_pd.save(path=aggregated_results_valid_path, df=results_valid_df)

    clean_and_save_results(
        run_name=version_name,
        out_path_prefix='results_preprocessed',
        file_prefix=aggregated_results_name[:-4],
        results_dir_input=output_path,
        results_dir_output=output_path,
        constraints=None,
        run_name_in_input_path=False,
        run_name_in_output_path=False,
    )

    print(f'Saved to {aggregated_results_path}!')
    if aggregate_leaderboard:
        aggregated_leaderboard_name = f'leaderboard{constraint_str}{version_name_str}.csv'
        aggregated_leaderboard_path = f'{output_path}{aggregated_leaderboard_name}'
        print(f'Saving output to "{aggregated_leaderboard_path}"')
        save_pd.save(path=aggregated_leaderboard_path, df=leaderboards_df)
        print(f'Success! Saved output to "{aggregated_leaderboard_path}"')

        clean_and_save_results(
            run_name=version_name,
            out_path_prefix='leaderboard_preprocessed',
            file_prefix=aggregated_leaderboard_name[:-4],
            results_dir_input=output_path,
            results_dir_output=output_path,
            constraints=None,
            run_name_in_input_path=False,
            run_name_in_output_path=False,
        )

    if aggregate_zeroshot:
        save_zs_metadata_per_task(
            unique_tasks_dict=None,
            banned_tasks=banned_tasks,
            results_valid_df=results_valid_df,
            output_suite_context=output_suite_context,
        )

        if leaderboards_df is not None:
            dataset_size_mb_df = get_dataset_size_mb(leaderboards_df=leaderboards_df, results_valid_df=results_valid_df, output_suite_context=output_suite_context)
            dataset_size_mb_name = f'dataset_zs_size{constraint_str}{version_name_str}.csv'
            dataset_size_mb_path = f'{output_path}{dataset_size_mb_name}'
            print(f'Saving output to "{dataset_size_mb_path}"')
            save_pd.save(path=dataset_size_mb_path, df=dataset_size_mb_df)
            print(f'Success! Saved output to "{dataset_size_mb_path}"')


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
    parser.set_defaults(version_name="2023_03_19_zs")  # FIXME: Remove
    parser.set_defaults(constraint="24h64c")  # FIXME: Remove
    args = parser.parse_args()

    version_name = args.version_name
    include_infer_speed = False
    keep_params = False

    VERSION_NAME_HACK = "2023_07_25"
    VERSION_NAME_HACK = "2023_08_21"
    version_name = VERSION_NAME_HACK

    path_prefix = f's3://{args.s3_bucket}/{args.s3_prefix}{version_name}/'

    # task_metadata_244 = load_task_metadata(path='task_metadata_244.csv')
    task_metadata_289 = load_task_metadata(path='task_metadata_289.csv')
    allowed_tids = set(list(task_metadata_289['tid']))

    output_path = f's3://{args.s3_bucket}/aggregated/ec2/{version_name}/'

    aggregate_all(
        path_prefix=path_prefix,
        version_name=version_name,
        # constraint=args.constraint,
        allowed_tids=allowed_tids,
        include_infer_speed=include_infer_speed,
        keep_params=keep_params,
        output_path=output_path,
        aggregate_leaderboard=True,
        aggregate_zeroshot=True,
        aggregate_model_failures=True,
        aggregate_logs=False,
        max_size_mb=10,
        mode=args.mode,
    )
