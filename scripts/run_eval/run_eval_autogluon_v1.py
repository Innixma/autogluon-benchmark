from autogluon.bench.eval.scripts.run_evaluation_openml import evaluate


if __name__ == '__main__':
    framework_name_suffix = '_1h8c_gp3_amlb_2023'
    frameworks_run = [
        'AutoGluon_benchmark',
        'AutoGluon_hq',
        'autosklearn',
        'flaml',
        'H2OAutoML',
        'lightautoml',
        # 'GAMA_benchmark',
        'mljarsupervised_benchmark',
        # 'mljarsupervised_perform',
        # 'TPOT',
        # 'TunedRandomForest',
        # 'RandomForest',
        # # 'NaiveAutoML',
    ]
    frameworks_run = [f + framework_name_suffix for f in frameworks_run]

    frameworks_run = [
        # 'AutoGluon_bq_DSL_4h8c_2023_11_03',
        # 'AutoGluon_bq_4h8c_2023_11_03',
        'AutoGluon_bq_DSL_1h8c_2023_11_03',
        # 'AutoGluon_bq_1h8c_2023_11_03',
        #
        # 'AutoGluon_bq_4h8c_2023_11_02',
        # 'AutoGluon_bq_DSL_4h8c_2023_11_02',
        # 'AutoGluon_bq_DS_4h8c_2023_11_02',

        # Final PR
        # "PR_AG_DS_4h8c_2023_11_01",
        # "PR_AG_DSL_4h8c_2023_11_01",
        # "AutoGluon_bq_4h8c_2023_11_01",
        # "PR_AG_DS_1h8c_2023_11_01",
        # "PR_AG_DSL_1h8c_2023_11_01",
        # "AutoGluon_bq_1h8c_2023_11_01",

        # Lennart Final
        # 'AG_bq_M1910B_DSL_4h8c_2023_10_12_purucker',  # Dynamic stacking limited (does not go over time limit, 1/4 for first fit)
        # 'AG_bq_M1910_memory_fix_4h8c_2023_10_12_purucker',  # Best quality, latest master branch
        # 'AG_bq_M1910B_DS_4h8c_2023_10_12_purucker',  # Dynamic stacking (x2 time limit)
        # "AG_bq_M1910B_stack_4h8c_2023_10_12_purucker",  # always using stacking
        # "AG_bq_M1910B_no_stack_4h8c_2023_10_12_purucker",  # never use stacking

        # 'AutoGluon_bq_DSL_4h8c_2023_11_02',
        # 'AutoGluon_bq_DS_4h8c_2023_11_02',

        # 'AutoGluon_bq_4h8c_2022_03_25',
        # 'AutoGluon_hq_4h8c_2022_03_25',



        # Current best
        # 'AG_DS_FL3_memory_fix_4h8c_2023_10_12_purucker',

        # Current baseline
        # 'AG_bq_4h8c_2023_10_12_purucker',  # without memory estimation fix


        # 'AG_ho_dynamic_stacking_4h8c_2023_10_12_purucker',
        # 'AG_ho_dynamic_stacking_limited_4h8c_2023_10_12_purucker',
        # 'AG_no_stack_4h8c_2023_10_12_purucker',

        # 'AG_stack_4h8c_2023_10_12_purucker',

        # "AG_ho_dynamic_stacking_4h8c_2023_09_25_infoleak",

        # "AG_stack_ho_dynamic_clean_oof_v2_4h8c_2023_09_28_purucker",
        # "AG_stack_ho_dynamic_clean_oof_4h8c_2023_09_28_purucker",
        # "AutoGluon_bq_4h8c_2023_08_21",

        # 'Ensemble_AG_FTT_all_bq_mytest24h_2022_09_14',

        # "AutoGluon_bq_clipping_4h8c_2023_10_06",
        # "AutoGluon_bq_sigmoid_4h8c_2023_10_06",

        # "AG_ho_dynamic_stacking_4h8c_2023_09_25_infoleak",

        # 'Ensemble_AG_FTT_all_bq_cpu_mytest24h_2022_09_14',
        # 'Ensemble_AG_bq_mytest24h_2022_09_14',
        # 'Ensemble_AG_FTT_all_bq_mytest4h_2022_09_14',
        # 'Ensemble_AG_bq_mytest4h_2022_09_14',

        # 'AG_no_stack_4h8c_2023_10_09_purucker',
        # 'AG_no_stack_1h8c_2023_10_09_purucker',
        #
        # 'AG_bq_4h8c_2023_10_09_purucker',
        # 'AG_bq_1h8c_2023_10_09_purucker',
        #
        # 'AG_stack_4h8c_2023_10_09_purucker',
        # 'AG_stack_1h8c_2023_10_09_purucker',

        # 'AG_ho_dynamic_stacking_4h8c_2023_10_09_purucker',
        # 'AG_ho_dynamic_stacking_1h8c_2023_10_09_purucker',

        # 'AG_ho_dynamic_stacking_limited_4h8c_2023_10_09_purucker',
        # 'AG_ho_dynamic_stacking_limited_1h8c_2023_10_09_purucker',


    ] + frameworks_run

    # frameworks_run = None

    frameworks_run = [
        # "AutoGluon_bq_N100_1h8c_2023_11_08_tabrepo",
        # "AutoGluon_bq_N100_4h8c_2023_11_08_tabrepo",
        # "AutoGluon_bq_bag8_N100_1h8c_2023_11_08_tabrepo",
        # "AutoGluon_bq_bag8_N100_4h8c_2023_11_08_tabrepo",
        "AutoGluon_bq_ds_N100_1h8c_2023_11_08_tabrepo",
        # "AutoGluon_bq_ds_N100_4h8c_2023_11_08_tabrepo",
        # "AutoGluon_bq_nodefaults_N100_1h8c_2023_11_08_tabrepo",
        # "AutoGluon_bq_nodefaults_N100_4h8c_2023_11_08_tabrepo",
        # "AutoGluon_bq_norepeats_N100_1h8c_2023_11_08_tabrepo",
        # "AutoGluon_bq_norepeats_N100_4h8c_2023_11_08_tabrepo",
        'AutoGluon_bestquality_1h8c_2021_02_06_v0_1_0',  # v0.1.0
        # 'AutoGluon_bestquality_1h_2021_09_02',  # v0.3.1
        "AutoGluon_benchmark_1h8c_gp3_amlb_2022",
    ] + frameworks_run

    paths = [
        "s3://automl-benchmark-ag/aggregated/amlb_purucker/2023_10_12_purucker/results_preprocessed.csv",
        # "s3://automl-benchmark-bingzzhu/aggregated/ec2/2022_09_14/results_preprocessed.csv",
        "s3://automl-benchmark-ag/aggregated/amlb/amlb_2023_preprocessed.csv",
        "s3://automl-benchmark-ag/aggregated/amlb/amlb_2022_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/amlb_purucker/2023_09_28_purucker/results_preprocessed2023_09_28_purucker.csv",
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_08_21/results_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_09_25_infoleak/results_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_10_06/results_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/amlb_purucker/2023_10_09_purucker/results_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_11_01/results_preprocessed.csv",
        "s3://automl-benchmark-ag/aggregated/ec2/2022_03_25/results_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_11_02/results_preprocessed.csv",
        "s3://automl-benchmark-ag/aggregated/ec2/2023_11_03/results_preprocessed.csv",
        "s3://automl-benchmark-ag/aggregated/ec2/2023_11_08_tabrepo/results_preprocessed.csv",
        "s3://automl-benchmark-ag/aggregated/ec2/2021_02_06_v0_1_0/results_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/ec2/2021_09_02/results_preprocessed.csv",
        # "s3://automl-benchmark-ag/aggregated/ec2/2023_11_10/leaderboard_preprocessed.csv"
    ]

    # frameworks_run = None

    problem_types = [
        'binary',
        'multiclass',
        'regression',
    ]
    # problem_types = None
    use_tid_as_dataset_name = False
    treat_folds_as_datasets = False
    task_metadata = "task_metadata.csv"
    infer_batch_size = None

    # banned_datasets = ['Bioresponse', 'KDDCup99', 'Satellite', 'adult', 'dionis', 'sf-police-incidents']
    banned_datasets = None
    banned_datasets = [
        # "Brazilian_houses", "cmc", "arcene", "kr-vs-kp"
        # 'OnlineNewsPopularity', 'topo_2_1', 'us_crime',
        # 'KDDCup09-Upselling', 'KDDCup99', 'MiniBooNE', 'ada', 'dionis', 'helena', 'kick', 'madeline', 'ozone-level-8hr', 'segment', 'sf-police-incidents',
        # 'wine-quality-white',
        # 'Bioresponse', 'Satellite', 'adult',
    ]

    frameworks_rename = {
        "AG_bq_M1910B_DSL_4h8c_2023_10_12_purucker": "AutoGluon v1.0 Preview (Best, 4h8c)",
        "AutoGluon_benchmark_4h8c_gp3_amlb_2023": "AutoGluon v0.8.2 (Best, 4h8c)",
        "AutoGluon_benchmark_1h8c_gp3_amlb_2023": "AutoGluon v0.8.2 (Best, 1h8c)",
        "AutoGluon_hq_4h8c_gp3_amlb_2023": "AutoGluon v0.8.2 (High, 4h8c)",
        "AutoGluon_hq_1h8c_gp3_amlb_2023": "AutoGluon v0.8.2 (High, 1h8c)",
        "AutoGluon_bq_4h8c_2022_03_25": "AutoGluon v0.4.0 (Best, 4h8c)",
        "AutoGluon_bq_ds_N100_4h8c_2023_11_08_tabrepo": "AutoGluon v1.0 Preview (Best+ZS, 4h8c)",
        "AutoGluon_bq_ds_N100_1h8c_2023_11_08_tabrepo": "AutoGluon v1.0 Preview (Best+ZS, 1h8c)",
        "AutoGluon_bestquality_1h8c_2021_02_06_v0_1_0": "AutoGluon v0.1.0 (Best, 1h8c)",
        "AutoGluon_benchmark_1h8c_gp3_amlb_2022": "AutoGluon v0.3.1 (Best, 1h8c)",
        # "AutoGluon_bq_1h8c_2023_11_03": "AutoGluon v1.0 Preview (Best, 1h8c)",
        "AutoGluon_bq_DSL_1h8c_2023_11_03": "AutoGluon v1.0 Preview (Best, 1h8c)",
        "flaml_1h8c_gp3_amlb_2023": "FLAML (2023, 1h8c)",
        "mljarsupervised_benchmark_1h8c_gp3_amlb_2023": "MLJAR (2023, 1h8c)",
        "H2OAutoML_1h8c_gp3_amlb_2023": "H2OAutoML (2023, 1h8c)",
        "lightautoml_1h8c_gp3_amlb_2023": "lightautoml (2023, 1h8c)",
        "autosklearn_1h8c_gp3_amlb_2023": "autosklearn (2023, 1h8c)",
    }
    # frameworks_rename = None

    output_prefix = "autogluon_v1"

    evaluate_kwargs = dict(
        paths=paths,
        frameworks_run=frameworks_run,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=False,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        task_metadata=task_metadata,
        banned_datasets=banned_datasets,
        frameworks_rename=frameworks_rename,
    )

    evaluate(
        problem_type=problem_types,
        output_suffix=f'{output_prefix}/1h8c/all',
        **evaluate_kwargs,
    )
    evaluate(
        problem_type=problem_types,
        output_suffix=f'{output_prefix}/1h8c_fillna/all',
        framework_nan_fill='constantpredictor_1h8c_gp3_amlb_2023',
        compute_z_score=False,
        **evaluate_kwargs,
    )
    for problem_type in problem_types:
        evaluate(
            problem_type=problem_type,
            output_suffix=f'{output_prefix}/1h8c/{problem_type}',
            **evaluate_kwargs,
        )
        evaluate(
            problem_type=problem_type,
            output_suffix=f'{output_prefix}/1h8c_fillna/{problem_type}',
            framework_nan_fill='constantpredictor_1h8c_gp3_amlb_2023',
            compute_z_score=False,
            **evaluate_kwargs,
        )

    from autogluon.common.utils.s3_utils import upload_s3_folder, upload_file
    upload_s3_folder(bucket="autogluon-zeroshot", prefix="autogluon_v1/", folder_to_upload="data/results/output/openml/autogluon_v1/")
    import shutil
    shutil.make_archive("results", 'zip', "data/results/output/openml/autogluon_v1")
    upload_file(file_name="results.zip", bucket="autogluon-zeroshot", prefix="autogluon_v1")
