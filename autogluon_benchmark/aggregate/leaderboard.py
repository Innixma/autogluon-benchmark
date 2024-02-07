import copy
import time

from autogluon.common.savers import save_pd

from autogluon.bench.eval.benchmark_context.output_suite_context import OutputSuiteContext


def aggregate_leaderboards(path_prefix: str, contains=None, keep_params=True, include_infer_speed=False, mode='seq'):
    output_suite_context = OutputSuiteContext(path=path_prefix,
                                              contains=contains,
                                              include_infer_speed=include_infer_speed,
                                              keep_params=keep_params,
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
