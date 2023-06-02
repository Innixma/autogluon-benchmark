from autogluon_benchmark.evaluation import evaluate_results
from autogluon_benchmark.evaluation.constants import TIME_INFER_S
from autogluon_benchmark.evaluation.evaluate_utils import compute_stderr_z_stat, compute_stderr_z_stat_bulk, compute_win_rate_per_dataset, graph_vs
from autogluon_benchmark.evaluation import BenchmarkEvaluator


def run(
    *,
    frameworks_run,
    paths,
    output_suffix='ag_full_v5/1h8c',
    framework_nan_fill=None,
    problem_type=None,
    folds_to_keep: list = None,
    compute_z_score=True,
    treat_folds_as_datasets=False,
    banned_datasets=None,
    infer_batch_size=None,
    clean_data=True,
    use_tid_as_dataset_name=True,
    filter_errors=False,  # If True, all dataset errors will be filtered out
):
    results_dir = 'data/results/'
    if folds_to_keep is None:
        folds_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    frameworks_compare_vs_all = []
    if len(frameworks_compare_vs_all) == 0:
        frameworks_compare_vs_all = [frameworks_run[0]]

    benchmark_evaluator = BenchmarkEvaluator(
        results_dir=results_dir,
        output_suffix=output_suffix,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        framework_nan_fill=framework_nan_fill,
        filter_errors=filter_errors,
    )

    results_raw = benchmark_evaluator.load_data(paths=paths,
                                                frameworks=frameworks_run,
                                                folds=folds_to_keep,
                                                clean_data=clean_data,
                                                problem_type=problem_type,
                                                banned_datasets=banned_datasets,
                                                infer_batch_size=infer_batch_size,
                                                treat_folds_as_datasets=treat_folds_as_datasets)

    folds_to_keep = sorted(results_raw['fold'].unique())

    if len(folds_to_keep) > 1:
        compute_win_rate_per_dataset(f1=frameworks_run[0], f2=frameworks_run[1], results_raw=results_raw, folds=folds_to_keep)
    if compute_z_score and len(folds_to_keep) > 1:
        z_stat_df = compute_stderr_z_stat_bulk(framework=frameworks_run[0], frameworks_to_compare=frameworks_run[1:], results_raw=results_raw)
        z_stat_series = compute_stderr_z_stat(results_raw, f1=frameworks_run[0], f2=frameworks_run[1], folds=folds_to_keep, verbose=False)
        graph_vs(results_df=results_raw, f1=frameworks_run[0], f2=frameworks_run[1], z_stats=z_stat_series)

    results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
        results_raw=results_raw,
        frameworks=frameworks_run,
        columns_to_agg_extra=[
            TIME_INFER_S,
        ],
        frameworks_compare_vs_all=frameworks_compare_vs_all,
        output_dir=benchmark_evaluator.results_dir_output,
    )


if __name__ == '__main__':
    frameworks_run = [
        # 'AutoGluon_bq_1h8c_2022_03_25',  # v0.4.0
        # 'AutoGluon_bq_1h8c_2022_05_09_rf',  # v0.4.1
        # 'AutoGluon_hq_1h8c_2022_03_25',  # v0.4.0
        # 'AutoGluon_bestquality_1h_2021_02_06_v0_1_0',  # v0.1.0
        # 'AutoGluon_bestquality_1h_2021_09_02',  # v0.3.1

        # 'H2OAutoML_1h8c_2022_03_25',
        # 'flaml_4h8c_gp3_2022_jmlr',

        # 'AutoGluon_bq_1h8c_2022_11_14_v06_is',
        # 'AutoGluon_hq_1h8c_2022_11_14_v06_is',
        # 'AutoGluon_gq_1h8c_2022_11_14_v06_is',
        # 'AutoGluon_mq_1h8c_2022_11_14_v06_is',

        'AutoGluon_bq_1h8c_2022_11_14_v06_is',
        'AutoGluon_bq_1h8c_2023_02_14_v07_infer_speed',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_infer_speed',

        # 'AutoGluon_mq_1h8c_2023_02_20_bool',
        # 'AutoGluon_mq_1h8c_2023_02_20_bool_LightGBM',
        # 'AutoGluon_mq_1h8c_2023_02_20_bool_test_LightGBM',
        # 'AutoGluon_mq_1h8c_2023_02_20_bool_test_CatBoost',
        # 'AutoGluon_bq_1h8c_2023_02_14_v07_sk102_infer_speed',

        # 'AutoGluon_hq_il001_1h8c_2022_11_14_v06_is',
        # 'AutoGluon_hq_il0005_1h8c_2022_11_14_v06_is',

        # 'AutoGluon_bq_1h8c_2023_02_12_v07_infer_speed',
        # 'AutoGluon_hq_1h8c_2023_02_12_v07_infer_speed',
        # 'AutoGluon_gq_1h8c_2023_02_12_v07_infer_speed',
        # 'AutoGluon_mq_1h8c_2023_02_12_v07_infer_speed',

        # 'AutoGluon_mq_1h8c_2023_02_13_v07_infer_speed',
        # 'AutoGluon_bq_1h8c_2023_02_14_v07_infer_speed',
        # 'AutoGluon_hq_1h8c_2023_02_14_v07_infer_speed',
        # 'AutoGluon_gq_1h8c_2023_02_14_v07_infer_speed',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_infer_speed',

        # 'AutoGluon_mq_1h8c_2022_11_14_v06_RandomForestEntr',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_RandomForestEntr',

        # 'AutoGluon_bq_1h8c_2022_11_14_v06_RandomForestEntr_BAG_L1',
        # 'AutoGluon_bq_1h8c_2023_02_14_v07_RandomForestEntr_BAG_L1',

        # 'AutoGluon_bq_1h8c_2022_11_14_v06_RandomForestGini_BAG_L1',
        # 'AutoGluon_bq_1h8c_2023_02_14_v07_RandomForestGini_BAG_L1',

        # 'AutoGluon_bq_1h8c_2022_11_14_v06_ExtraTreesEntr_BAG_L1',
        # 'AutoGluon_bq_1h8c_2023_02_14_v07_ExtraTreesEntr_BAG_L1',

        # 'AutoGluon_mq_1h8c_2022_11_14_v06_KNeighborsUnif',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_KNeighborsUnif',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_sk102_KNeighborsUnif',

        # 'AutoGluon_mq_1h8c_2022_11_14_v06_ExtraTreesEntr',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_ExtraTreesEntr',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_sk102_ExtraTreesEntr',

        # 'AutoGluon_mq_1h8c_2022_11_14_v06_RandomForestEntr',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_RandomForestEntr',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_sk102_RandomForestEntr',
        #
        # 'AutoGluon_mq_1h8c_2022_11_14_v06_CatBoost',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_CatBoost',
        # 'AutoGluon_mq_1h8c_2023_02_20_bool_CatBoost',
        #
        # 'AutoGluon_mq_1h8c_2022_11_14_v06_XGBoost',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_XGBoost',
        #
        # 'AutoGluon_mq_1h8c_2022_11_14_v06_LightGBM',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_LightGBM',
        # 'AutoGluon_mq_1h8c_2023_02_20_bool_LightGBM',
        #
        # 'AutoGluon_mq_1h8c_2022_11_14_v06_NeuralNetTorch',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_NeuralNetTorch',
        #
        # 'AutoGluon_mq_1h8c_2022_11_14_v06_NeuralNetFastAI',
        # 'AutoGluon_mq_1h8c_2023_02_14_v07_NeuralNetFastAI',

        # 'AutoGluon_bq_1h8c_2022_11_14_v06_NeuralNetFastAI_BAG_L1',
        # 'AutoGluon_bq_1h8c_2023_02_14_v07_NeuralNetFastAI_BAG_L1',

        # 'AutoGluon_bq_1h8c_2023_01_08_v062',
        # 'AutoGluon_bq_1h8c_2022_12_08_v061',
        # 'AutoGluon_bestquality_1h8c_2022_05_09',
        # 'AutoGluon_hq_1h8c_2023_01_08_v062',
        # 'AutoGluon_gq_1h8c_2023_01_08_v062',
        # 'AutoGluon_mq_1h8c_2023_01_08_v062',
        # 'AutoGluon_hq_1h8c_2022_12_08_v061',
        # 'AutoGluon_gq_1h8c_2022_12_08_v061',
        # 'AutoGluon_mq_1h8c_2022_12_08_v061',

        # 'EnsembleTrueCV',
        # 'EnsembleAllHPO',
        # 'EnsembleAGSimple',
        # 'AutoGluon_bq_4h64c_2022_11_14_v06_ftt',
        # 'AutoGluon_hq_4h64c_2022_11_14_v06_ftt',
        # 'AutoGluon_ebq_4h64c_2022_11_14_v06_ftt',
        # 'AutoGluon_ehq_4h64c_2022_11_14_v06_ftt',
        # 'AutoGluon_euq_4h64c_2022_11_14_v06_ftt',

        # 'Ensemble_AG_FTT_all_bq_mytest24h_2022_09_14_v3',
        # 'Ensemble_AG_FTT_all_bq_cpu_mytest24h_2022_09_14_v3',
        # 'Ensemble_AG_bq_mytest24h_2022_09_14',
        # 'Ensemble_AG_FTT_all_bq_mytest4h_2022_09_14_v2',
        # 'Ensemble_AG_bq_mytest4h_2022_09_14_v2',
        # 'AutoGluon_bq_1h8c_2022_06_26_binary',
        # 'AutoGluon_hq_1h8c_2022_06_26_binary',
        # 'AutoGluon_gq_1h8c_2022_06_26_binary',
        # 'AutoGluon_mq_1h8c_2022_06_26_binary',

        # 'AutoGluon_benchmark_1h8c_gp3_2022_jmlr',
        # 'autosklearn_4h8c_gp3_2022_jmlr',
        # 'autosklearn2_1h8c_gp3_2022_jmlr',
        # 'flaml_1h8c_gp3_2022_jmlr',
        # 'GAMA_benchmark_1h8c_gp3_2022_jmlr',
        # 'H2OAutoML_1h8c_gp3_2022_jmlr',
    ]

    paths = [
        'openml_ag_2021_02_06_v0_1_0.csv',  # 10-fold ag 1h
        'openml_ag_2021_09_02.csv',  # 0.3.1
        'openml_ag_2022_03_25.csv',  # 1h8c + 4h8c mq gq hq bq AG 0.4 PyPi + other frameworks
        'openml_ag_2022_05_09.csv',  # 1h8c + 4h8c mq bq 0.4.1
        'openml_ag_2022_06_26_binary.csv',  # 1h8c bq hq gq mq with binary no stack
        'openml_ag_2022_06_26_binary_models.csv',  # 1h8c bq hq gq mq with binary no stack
        'amlb/2022_jmlr.csv',  # gjibers et al

        'openml_ag_2022_09_14.csv',  # Bingzhao
        'openml_ag_2022_09_14_v2.csv',
        'openml_ag_2022_09_14_v3.csv',

        'openml_ag_2022_11_14_v06_ftt.csv',  # 4h64c ebq, ehq, eup, bq, hq, 1fold
        'openml_ag_2022_11_14_v06.csv',  # 1h8c bq, hq, gq, mq, hqi001 hqi0005 hqi0002
        'openml_ag_2022_11_14_v06_is.csv',

        'zeroshot/zeroshot_EnsembleTrueCV.csv',
        'zeroshot/zeroshot_EnsembleAllHPO.csv',
        'zeroshot/zeroshot_EnsembleAGSimple.csv',
        'openml_ag_2022_10_13_zs_models.csv',

        'openml_ag_2022_12_08_v061.csv',
        'openml_ag_2023_01_08_v062.csv',

        'openml_ag_2023_02_12_v07_infer_speed.csv',
        'openml_ag_2023_02_13_v07_infer_speed.csv',  # Accelerated FastAI Preprocessing
        'openml_ag_2023_02_14_v07_infer_speed.csv',  # Fixed TorchNN time_limit

        'openml_ag_2023_02_14_v07_models.csv',
        'openml_ag_2022_11_14_v06_models.csv',

        'openml_ag_2023_02_14_v07_sk102_infer_speed.csv',
        'openml_ag_2023_02_14_v07_sk102_models.csv',

        'openml_ag_2023_02_20_bool.csv',
        'openml_ag_2023_02_20_bool_models.csv',

        'openml_ag_2023_02_20_bool_test_models.csv',

    ]

    use_tid_as_dataset_name = False
    # problem_types = ['multiclass']
    # problem_types = ['binary', 'regression']
    problem_types = ['binary', 'multiclass', 'regression']
    treat_folds_as_datasets = False
    folds_to_keep = [0]
    # infer_batch_size = 1
    filter_errors = True
    infer_batch_size = None
    banned_datasets = [
        'car',
        'kr-vs-kp',
        'OnlineNewsPopularity',
    ]
    # problem_types = ['multiclass']
    # for problem_type in problem_types:
    #     run(
    #         output_suffix=f'2022_amlb_jmlr/1h8c/{problem_type}',
    #         problem_type=problem_type,
    #     )
    #     run(
    #         output_suffix=f'2022_amlb_jmlr/1h8c_fillna/{problem_type}',
    #         problem_type=problem_type,
    #         framework_nan_fill='constantpredictor_1h8c_gp3_2022_jmlr',
    #     )
    run(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=f'2022_amlb_jmlr/1h8c/all',
        problem_type=problem_types,
        treat_folds_as_datasets=treat_folds_as_datasets,
        # folds_to_keep=folds_to_keep,
        infer_batch_size=infer_batch_size,
        filter_errors=filter_errors,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        banned_datasets=banned_datasets,
        # folds_to_keep=folds_to_keep,
    )
    run(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=f'2022_amlb_jmlr/1h8c_fillna/all',
        framework_nan_fill='constantpredictor_1h8c_gp3_2022_jmlr',
        problem_type=problem_types,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=filter_errors,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        banned_datasets=banned_datasets,
        # folds_to_keep=folds_to_keep,
        compute_z_score=False,
    )
