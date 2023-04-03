from autogluon_benchmark.evaluation import evaluate_results
from autogluon_benchmark.evaluation.constants import TIME_INFER_S
from autogluon_benchmark.evaluation import BenchmarkEvaluator


def run(
        *,
        frameworks_run=None,
        output_suffix='ag_full_v6/1h8c',
        framework_nan_fill=None,
        problem_type=None,
):
    results_dir = 'data/results/'
    s3_input_dir = 's3://autogluon-zeroshot/benchmark_baselines/results/input/prepared/openml'

    paths = [
        # 'openml_ag_2022_06_19.csv',  # 1h8c bq hq gq mq 0.5 pre
        # 'openml_ag_2022_06_19_nn.csv',  # 1h8c bq hq gq mq 0.5 pre with nn num_cpus opt
        # 'openml_ag_2022_06_19_models.csv',  # 1h8c bq hq gq mq 0.5 pre
        # 'openml_ag_2022_06_19_nn_models.csv',  # 1h8c bq hq gq mq 0.5 pre with nn num_cpus opt
        # 'openml_ag_2022_06_21.csv',  # 4h64c bq hq gq mq 0.5 pre
        # 'openml_ag_2022_06_21_nn.csv',  # 4h64c bq hq gq mq 0.5 pre with nn num_cpus opt
        # 'openml_ag_2022_06_21_models.csv',  # 4h64c bq hq gq mq 0.5 pre
        # 'openml_ag_2022_06_21_nn_models.csv',  # 4h64c bq hq gq mq 0.5 pre with nn num_cpus opt
        # 'openml_ag_2022_06_25_nn.csv',  # 4h64c bq mq 0.5 pre with nn num_cpus opt
        # 'openml_ag_2022_06_25_nn_models.csv',  # 4h64c bq mq 0.5 pre with nn num_cpus opt
        # 'openml_ag_2022_06_26_binary.csv',  # 1h8c bq hq gq mq with binary no stack
        # 'openml_ag_2022_06_26_binary_models.csv',  # 1h8c bq hq gq mq with binary no stack
        # 'amlb/2022_jmlr.csv',  # gjibers et al
        # 'openml_ag_2022_07_12_torch.csv',  # 1h8c bq hq gq mq torch 1.12
        # 'openml_ag_2022_07_12_torch_models.csv',  # 1h8c bq hq gq mq torch 1.12
        # 'openml_ag_2022_07_26_i001.csv',
        # 'openml_ag_2022_07_26_i001_2.csv',
        # 'openml_ag_2022_09_30_gbm_zs_models.csv',
        # 'openml_ag_2022_09_30_cat_zs_models.csv',
        # 'openml_ag_2022_10_02_zs_models.csv',
        # 'openml_ag_2022_10_05_zs_models.csv',
        # 'zeroshot/zeroshot_lightgbm.csv',
        # 'zeroshot/zeroshot_all.csv',
        # 'zeroshot/zeroshot_all_v2.csv',
        # 'zeroshot/zeroshot_all_v2_25.csv',
        # 'zeroshot/zeroshot_all_v3_20.csv',
        # 'zeroshot/zeroshot_all_v4_14.csv',
        # 'zeroshot/zeroshot_all_v4_20.csv',
        # 'zeroshot/zeroshot_all_v4_32.csv',
        # 'zeroshot/zeroshot_all_rf_v4_7.csv',
        # 'zeroshot/zeroshot_all_rf_v4_1fold_10.csv',
        # 'zeroshot/zeroshot_all_rf_v4_9fold_7.csv',
        # 'zeroshot/zeroshot_all_CV.csv',
        # 'zeroshot/zeroshot_all_CV_10.csv',
        # 'zeroshot/zeroshot_v2_CV2.csv',
        # 'zeroshot/zeroshot_v2_CV2_5.csv',
        # 'zeroshot/zeroshot_v2_CV2_10_VS.csv',
        # 'zeroshot/zeroshot_v2_CV5_20_VS.csv',
        # 'zeroshot/ag_sim_v1.csv',
        # 'zeroshot/ag_sim_v1_custom.csv',
        # 'openml_ag_2022_10_04_zs.csv',
        # 'openml_ag_2022_10_13_zs_models.csv',
        #
        # 'openml_ag_2022_09_14.csv',  # Bingzhao
        # 'openml_ag_2022_09_14_v2.csv',  # Bingzhao with BQ
        # 'openml_ag_2022_09_14_v3.csv',  # Bingzhao with BQ
        #
        # 'openml_ag_2022_12_11_zs_models.csv',

        'openml_ag_2022_09_14.csv',  # Bingzhao
        'openml_ag_2022_09_14_v2.csv',  # Bingzhao with BQ
        'openml_ag_2022_09_14_v3.csv',  # Bingzhao with BQ

    ]

    paths_ensbag289 = [
        f's3://autogluon-zeroshot/config_results/zs_EnsBag289_C608_F3_CV_S{i+1}.csv' for i in range(40)
    ]

    paths = paths + paths_ensbag289

    use_tid_as_dataset_name = False
    clean_data = True

    benchmark_evaluator = BenchmarkEvaluator(
        results_dir=results_dir,
        output_suffix=output_suffix,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        filter_errors=True,
        task_metadata='task_metadata_244.csv',
    )

    folds_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    frameworks_run = [
        # 'AutoGluon_benchmark_1h8c_gp3_2022_jmlr',
        # 'autosklearn_1h8c_gp3_2022_jmlr',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_LightGBMXT',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_LightGBMLarge',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_NeuralNetFastAI',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_NeuralNetTorch',
        # 'H2OAutoML_1h8c_gp3_2022_jmlr',
        # 'AutoGluon_zs_GBM_8h8c_2022_09_30_gbm_zs_LightGBM_c1',
        # 'EnsembleTrueCV',
        # 'EnsembleCV',
        # 'SingleBestCV',
        #
        # 'AutoGluon_mq_4h64c_2022_06_21_XGBoost',
        # 'AutoGluon_mq_4h64c_2022_06_21_nn_XGBoost',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_XGBoost',
        #
        # 'AutoGluon_mq_4h64c_2022_06_21_RandomForestEntr',
        # 'AutoGluon_mq_4h64c_2022_06_21_RandomForestGini',
        # 'AutoGluon_mq_4h64c_2022_06_21_ExtraTreesEntr',
        # 'AutoGluon_mq_4h64c_2022_06_21_ExtraTreesGini',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_RandomForestEntr',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_RandomForestGini',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_ExtraTreesEntr',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_ExtraTreesGini',

        # 'AutoGluon_mq_4h64c_2022_06_21_WeightedEnsemble_L2',
        # 'AutoGluon_mq_4h64c_2022_06_21_RandomForestEntr',
        # 'AutoGluon_mq_4h64c_2022_06_21_RandomForestGini',
        # 'AutoGluon_mq_4h64c_2022_06_21_ExtraTreesEntr',
        # 'AutoGluon_mq_4h64c_2022_06_21_ExtraTreesGini',

        # 'Ensemble_AG_FTT_all_bq_mytest24h_2022_09_14_v3',
        # 'Ensemble_AG_FTT_all_bq_mytest24h_2022_09_14',
        # 'Ensemble_AG_bq_mytest24h_2022_09_14',

        # 'AutoGluon_mq_4h64c_2022_06_21_XGBoost',
        # 'AutoGluon_mq_4h64c_2022_06_21_autogluon_single',
        # 'AutoGluon_mq_4h64c_2022_06_21_CatBoost',
        # 'AutoGluon_mq_4h64c_2022_06_21_LightGBM',
        # 'AutoGluon_mq_4h64c_2022_06_21_LightGBMXT',
        # 'AutoGluon_mq_4h64c_2022_06_21_LightGBMLarge',
        # 'AutoGluon_mq_4h64c_2022_06_21_NeuralNetFastAI',
        # 'AutoGluon_mq_4h64c_2022_06_21_NeuralNetTorch',
        # 'AutoGluon_mq_4h64c_2022_06_21',
        # 'AutoGluon_mq_4h64c_2022_06_21_KNeighborsDist',
        # 'AutoGluon_mq_4h64c_2022_06_21_KNeighborsUnif',

        # 'AutoGluon_mq_4h64c_2022_06_21_NeuralNetFastAI',
        # 'AutoGluon_mq_4h64c_2022_06_21_NeuralNetTorch',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_NeuralNetFastAI',
        # 'AutoGluon_mq_4h64c_2022_06_25_nn_NeuralNetTorch',
        # 'ZS_FULL_CV_2',
        # 'ZS_FULL5_CV_2',
        # 'ag_sim_v1',
        # 'ag_sim_v1_custom',



        # 'AutoGluon_mq_4h64c_2022_06_21',
        # 'Ensemble_AG_FTT_all_bq_mytest4h_2022_09_14_v2',
        # 'Ensemble_AG_bq_mytest4h_2022_09_14_v2',

        # 'Ensemble_AG_FTT_all_mq_mytest4h_2022_09_14_v2',
        # 'Ensemble_AG_FTT_pretrain_mytest4h_2022_09_14',
        # 'Ensemble_AG_mq_mytest4h_2022_09_14_v2',
        # 'Ensemble_AG_FastFTT_mytest4h_2022_09_14',
        # 'Ensemble_AG_FTT_mytest4h_2022_09_14',
        #


        # 'Ensemble_AG_FTT_rowatt_mytest4h_2022_09_14',

        # 'AutoGluon_benchmark_4h8c_gp3_2022_jmlr',
        # 'AutoGluon_bq_1h8c_2022_06_26_binary',
        # 'AutoGluon_hq_1h8c_2022_06_26_binary',
        # 'AutoGluon_gq_1h8c_2022_06_26_binary',
        # 'AutoGluon_mq_1h8c_2022_06_26_binary',

        # 'flaml_4h8c_gp3_2022_jmlr',
        # 'autosklearn_4h8c_gp3_2022_jmlr',
        # 'GAMA_benchmark_4h8c_gp3_2022_jmlr',
        # 'H2OAutoML_4h8c_gp3_2022_jmlr',
        # 'lightautoml_4h8c_gp3_2022_jmlr',
        # 'mljarsupervised_benchmark_4h8c_gp3_2022_jmlr',

        # 'AutoGluon_benchmark_1h8c_gp3_2022_jmlr',
        # 'autosklearn_1h8c_gp3_2022_jmlr',
        # 'autosklearn2_1h8c_gp3_2022_jmlr',
        # 'flaml_1h8c_gp3_2022_jmlr',
        # 'GAMA_benchmark_1h8c_gp3_2022_jmlr',
        # 'H2OAutoML_1h8c_gp3_2022_jmlr',
        # 'lightautoml_1h8c_gp3_2022_jmlr',
        # 'mljarsupervised_benchmark_1h8c_gp3_2022_jmlr',
        # 'TPOT_1h8c_gp3_2022_jmlr',
        # 'TunedRandomForest_1h8c_gp3_2022_jmlr',
        # 'RandomForest_1h8c_gp3_2022_jmlr',
        # 'AutoGluon_mq_4h64c_2022_06_21_XGBoost',
        # 'AutoGluon_mq_4h64c_2022_06_21_CatBoost',
        # 'AutoGluon_mq_4h64c_2022_06_21_LightGBM',
        # 'AutoGluon_mq_4h64c_2022_06_21_NeuralNetFastAI',
        # 'AutoGluon_mq_4h64c_2022_06_21_NeuralNetTorch',
        # 'ALL_zs_v2_25',
        # 'AutoGluon_zs_v1_4h64c_2022_10_04_zs',
        # 'AutoGluon_zs_v1_mq_4h64c_2022_10_04_zs',
        # 'AutoGluon_zs_v1_hq_4h64c_2022_10_04_zs',
        # 'AutoGluon_zs_v1_bq_4h64c_2022_10_04_zs',
        # 'AutoGluon_mq_4h64c_2022_10_04_zs',
        # 'AutoGluon_gq_4h64c_2022_10_04_zs',
        # 'AutoGluon_hq_4h64c_2022_10_04_zs',
        # 'AutoGluon_bq_4h64c_2022_10_04_zs',
        # 'AutoGluon_zs_v1_4h8c_2022_10_04_zs',
        # 'ALL_zs_rf_v4_7',
        # 'ALL_zs_rf_v4_1fold_10',
        # 'ALL_zs_rf_v4_9fold_7',
        # 'ZS_TEST_CV_10',
        # 'ZS_FULL20_CV_5_VS',
        # 'ZS_FULL10_CV_2_VS',
        # 'ZS_TEST_CV',
        # 'ALL_zs_v2',
        # 'ALL_zs_v2_20',
        # 'ALL_zs_v4_14',
        # 'ALL_zs_v4_20',
        # 'ALL_zs_v4_32',
        # 'ALL_zs_v1',
        # 'LightGBM_zs_v1',
    ]

    zero_shot_runs = [
        ['AutoGluon_zs_bag_RF_16h64c_2022_12_11_zs_RandomForest', 1, 50],
        ['AutoGluon_zs_bag_XT_16h64c_2022_12_11_zs_ExtraTrees', 1, 50],
        ['AutoGluon_zs_bag_GBM_16h64c_2022_12_11_zs_LightGBM', 4, 200],
        ['AutoGluon_zs_bag_FASTAI_16h64c_2022_12_11_zs_NeuralNetFastAI', 1, 200],
        ['AutoGluon_zs_bag_CAT_16h64c_2022_12_11_zs_CatBoost', 1, 100],
    ]

    bagged_zs = True
    if bagged_zs:
        zs_suffix = '_BAG_L1'
    else:
        zs_suffix = ''
    frameworks_run_zeroshot = []
    for zero_shot_run, c_count, r_count in zero_shot_runs:

        for i in range(c_count):
            frameworks_run_zeroshot.append(f'{zero_shot_run}_c{i+1}{zs_suffix}')
        for i in range(r_count):
            frameworks_run_zeroshot.append(f'{zero_shot_run}_r{i+1}{zs_suffix}')
    frameworks_run_zeroshot = frameworks_run_zeroshot[:10]
    frameworks_run_zeroshot_rename = {f: f.rsplit('_zs_', 1)[-1] for f in frameworks_run_zeroshot}

    results_raw = benchmark_evaluator.load_data(
        paths=paths,
        frameworks=frameworks_run + frameworks_run_zeroshot,
        folds=folds_to_keep,
        clean_data=clean_data,
        problem_type=problem_type,
        banned_datasets=None,
        infer_batch_size=None,
        treat_folds_as_datasets=True,
    )

    results_raw['framework'] = results_raw['framework'].map(frameworks_run_zeroshot_rename).fillna(results_raw['framework'])
    frameworks_run_zeroshot = [frameworks_run_zeroshot_rename[f] for f in frameworks_run_zeroshot]
    frameworks_run += frameworks_run_zeroshot

    frameworks_compare_vs_all = []
    if len(frameworks_compare_vs_all) == 0:
        frameworks_compare_vs_all = [frameworks_run[0]]

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
    import time
    ts = time.time()
    problem_types = ['binary', 'multiclass', 'regression']
    run(
        output_suffix=f'2022_amlb_jmlr/1h8c/all',
    )
    te = time.time()
    print(f'Time Taken to Evaluate: {te-ts:.2f}s')
