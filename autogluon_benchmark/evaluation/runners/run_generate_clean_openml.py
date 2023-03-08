
import argparse
import pandas as pd

from autogluon.common.savers import save_pd

from autogluon_benchmark.evaluation.preprocess import preprocess_openml
from autogluon_benchmark.evaluation.constants import FRAMEWORK


def clean_and_save_results(
        run_name,
        results_dir='data/results/',
        results_dir_input=None,
        results_dir_output=None,
        file_prefix='results_automlbenchmark',
        run_name_in_input_path=True,
        constraints=None,
        out_path_prefix='openml_ag_',
        out_path_suffix='',
        framework_suffix_column='constraint',
):
    if results_dir_input is None:
        results_dir_input = results_dir + 'input/raw/'
    if results_dir_output is None:
        results_dir_output = results_dir + 'input/prepared/openml/'
    run_name_str = f'_{run_name}' if run_name_in_input_path else ''

    results_list = []
    if constraints is None:
        constraints = [None]
    for constraint in constraints:
        constraint_str = f'_{constraint}' if constraint is not None else ''
        results = preprocess_openml.preprocess_openml_input(
            path=results_dir_input + f'{file_prefix}{constraint_str}{run_name_str}.csv',
            framework_suffix=constraint_str,
            framework_suffix_column=framework_suffix_column,
        )
        results_list.append(results)

    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    if 'framework_parent' in results_raw.columns:
        results_raw[FRAMEWORK] = results_raw['framework_parent'] + '_' + run_name + '_' + results_raw[FRAMEWORK]
    else:
        results_raw[FRAMEWORK] = results_raw[FRAMEWORK] + '_' + run_name

    save_path = results_dir_output + f'{out_path_prefix}{run_name}{out_path_suffix}.csv'
    save_pd.save(path=save_path, df=results_raw)
    print(f'Saved file: {save_path}')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('--run_name', type=str, help="Name of run", nargs='?')
    # parser.add_argument('--file_prefix', type=str, help='Prefix of filename', default='results_automlbenchmark', nargs='?')
    # parser.add_argument('--constraints', type=list, help='Time constraints', default=None, nargs='?')
    # parser.add_argument('--out_path_suffix', type=str, help='Suffix added to output file name', default='', nargs='?')
    #
    # args = parser.parse_args()
    #
    # clean_and_save_results(
    #     run_name=args.run_name,
    #     file_prefix=args.file_prefix,
    #     constraints=args.constraints,
    #     out_path_suffix=args.out_path_suffix,
    # )

    # run_name_arg = '2022_06_19'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_06_21'
    # clean_and_save_results(run_name_arg, constraints=['4h64c'])
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['4h64c'],
    #     out_path_suffix='_models'
    # )
    # run_name_arg = '2022_07_26_i001'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_07_26_i001_2'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_07_26_saveall'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_06_26_binary'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_07_30'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_07_31_i01'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_07_31_i005'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_07_31_i002'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_09_30_gbm_zs'
    # # clean_and_save_results(run_name_arg, constraints=['8h8c'])
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['8h8c'],
    #     out_path_suffix='_models'
    # )
    # run_name_arg = '2022_09_30_cat_zs'
    # # clean_and_save_results(run_name_arg, constraints=['16h8c'])
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['16h8c'],
    #     out_path_suffix='_models'
    # )
    # run_name_arg = '2022_10_02_zs'
    # # clean_and_save_results(run_name_arg, constraints=['16h8c'])
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['16h8c'],
    #     out_path_suffix='_models'
    # )
    # run_name_arg = '2022_07_12_torch'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2022_09_14_v3'
    # clean_and_save_results(run_name_arg, constraints=['mytest24h'])

    # run_name_arg = '2022_11_14_v06_ftt'
    # clean_and_save_results(run_name_arg, constraints=['4h64c'])
    # run_name_arg = '2023_01_08_v062'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])
    # run_name_arg = '2023_02_14_v07_infer_speed'
    # clean_and_save_results(run_name_arg, constraints=['1h8c'])

    run_name_arg = '2023_02_27_zs'
    path_prefix = f's3://automl-benchmark-ag/aggregated/ec2/2023_02_27_zs/'
    clean_and_save_results(run_name_arg,
                           file_prefix='results',
                           results_dir_input=path_prefix,
                           run_name_in_input_path=False,
                           )
    clean_and_save_results(
        run_name_arg,
        results_dir_input=path_prefix,
        file_prefix='leaderboard',
        out_path_suffix='_models',
        run_name_in_input_path=False,
    )

    run_name_arg = '2023_02_27_zs'
    path_prefix = f's3://automl-benchmark-ag/aggregated/ec2/2023_02_27_zs/'
    clean_and_save_results(run_name_arg,
                           file_prefix='results',
                           results_dir_input=path_prefix,
                           run_name_in_input_path=False,
                           )
    clean_and_save_results(
        run_name_arg,
        results_dir_input=path_prefix,
        file_prefix='leaderboard',
        out_path_suffix='_models',
        run_name_in_input_path=False,
    )

    # run_name_arg = '2023_02_14_v07'
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['1h8c'],
    #     out_path_suffix='_models'
    # )
    # run_name_arg = '2023_02_14_v07_sk102'
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['1h8c'],
    #     out_path_suffix='_models'
    # )
    # run_name_arg = '2022_jmlr'
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='amlb_jmlr_2022/amlb',
    #     constraints=['all'],
    #     out_path_prefix='amlb/',
    #     framework_suffix_column='constraint'
    # )

    # run_name_arg = '2022_10_13_zs'
    # # clean_and_save_results(run_name_arg, constraints=['16h8c'])
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['16h8c'],
    #     out_path_suffix='_models'
    # )
    # run_name_arg = '2022_06_26_binary'
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['1h8c'],
    #     out_path_suffix='_models'
    # )
    #
    # run_name_arg = '2022_06_26_binary'
    # clean_and_save_results(
    #     run_name_arg,
    #     file_prefix='results_ag_leaderboard',
    #     constraints=['1h8c'],
    #     out_path_suffix='_models'
    # )

