from autogluon.bench.eval.scripts.run_evaluation_openml import evaluate


if __name__ == '__main__':
    constraint = "4h8c"

    framework_name_suffix = f'_{constraint}_gp3_amlb_2023'
    frameworks_run = [
        'AutoGluon_benchmark',
        'AutoGluon_hq',
        'autosklearn',
        'flaml',
        'H2OAutoML',
        'lightautoml',
        'GAMA_benchmark',
        'mljarsupervised_benchmark',
        # 'mljarsupervised_perform',
        # 'TPOT',
        # 'TunedRandomForest',
        # 'RandomForest',
        # # 'NaiveAutoML',
        # 'constantpredictor',
    ]
    frameworks_run = [f + framework_name_suffix for f in frameworks_run]

    frameworks_run += [
        # 'AutoGluon_bestquality_1h_2021_09_02',  # v0.3.1
        # 'AutoGluon_bestquality_1h8c_2021_02_06_v0_1_0',  # v0.1.0
        # "AutoGluon_benchmark_4h8c_gp3_amlb_2022",
    ]

    paths = [
        "s3://automl-benchmark-ag/aggregated/amlb/amlb_2023_preprocessed_min.parquet",
        # "s3://automl-benchmark-ag/aggregated/amlb/amlb_2022_preprocessed.csv",
        "s3://automl-benchmark-ag/aggregated/ec2/2023_11_20/results_preprocessed_min.parquet",
    ]

    frameworks_run = [
        # "AutoGluon_hq_simple_1h8c_2023_11_15",
        # "AutoGluon_bq_v1_1h8c_2023_11_20",
        # "AutoGluon_gq_v1_1h8c_2023_11_20",
        # "AutoGluon_hq_v1_1h8c_2023_11_20",
        # "AutoGluon_hq_v2_1h8c_2023_11_20",

        "AutoGluon_bq_v1_4h8c_2023_11_20",
        "AutoGluon_gq_v1_4h8c_2023_11_20",
        # "AutoGluon_hq_v1_4h8c_2023_11_20",
        "AutoGluon_hq_v2_4h8c_2023_11_20",

    ] + frameworks_run

    # frameworks_run = None

    problem_types = [
        'binary',
        'multiclass',
        'regression',
    ]
    use_tid_as_dataset_name = False
    treat_folds_as_datasets = False
    task_metadata = "task_metadata.csv"
    infer_batch_size = None

    banned_datasets = None

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

        "AutoGluon_bq_v1_1h8c_2023_11_20": "AutoGluon v1.0 (Best, 1h8c)",
        "AutoGluon_gq_v1_1h8c_2023_11_20": "AutoGluon v1.0 (High, 1h8c)",
        "AutoGluon_hq_v2_1h8c_2023_11_20": "AutoGluon v1.0 (Good, 1h8c)",
        "flaml_1h8c_gp3_amlb_2023": "FLAML (2023, 1h8c)",
        "mljarsupervised_benchmark_1h8c_gp3_amlb_2023": "MLJAR (2023, 1h8c)",
        "H2OAutoML_1h8c_gp3_amlb_2023": "H2OAutoML (2023, 1h8c)",
        "lightautoml_1h8c_gp3_amlb_2023": "lightautoml (2023, 1h8c)",
        "autosklearn_1h8c_gp3_amlb_2023": "autosklearn (2023, 1h8c)",
        "GAMA_benchmark_1h8c_gp3_amlb_2023": "GAMA (2023, 1h8c)",
        "TPOT_1h8c_gp3_amlb_2023": "TPOT (2023, 1h8c)",
        "TunedRandomForest_1h8c_gp3_amlb_2023": "TunedRandomForest (2023, 1h8c)",
        "RandomForest_1h8c_gp3_amlb_2023": "RandomForest (2023, 1h8c)",
        "constantpredictor_1h8c_gp3_amlb_2023": "constantpredictor (2023, 1h8c)",

        "AutoGluon_bq_v1_4h8c_2023_11_20": "AutoGluon v1.0 (Best, 4h8c)",
        "AutoGluon_gq_v1_4h8c_2023_11_20": "AutoGluon v1.0 (High, 4h8c)",
        "AutoGluon_hq_v2_4h8c_2023_11_20": "AutoGluon v1.0 (Good, 4h8c)",
        "flaml_4h8c_gp3_amlb_2023": "FLAML (2023, 4h8c)",
        "mljarsupervised_benchmark_4h8c_gp3_amlb_2023": "MLJAR (2023, 4h8c)",
        "H2OAutoML_4h8c_gp3_amlb_2023": "H2OAutoML (2023, 4h8c)",
        "lightautoml_4h8c_gp3_amlb_2023": "lightautoml (2023, 4h8c)",
        "autosklearn_4h8c_gp3_amlb_2023": "autosklearn (2023, 4h8c)",
        "GAMA_benchmark_4h8c_gp3_amlb_2023": "GAMA (2023, 4h8c)",
        "TPOT_4h8c_gp3_amlb_2023": "TPOT (2023, 4h8c)",
        "TunedRandomForest_4h8c_gp3_amlb_2023": "TunedRandomForest (2023, 4h8c)",
        "RandomForest_4h8c_gp3_amlb_2023": "RandomForest (2023, 4h8c)",
        "constantpredictor_4h8c_gp3_amlb_2023": "constantpredictor (2023, 4h8c)",
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
        output_suffix=f'{output_prefix}/{constraint}/all',
        **evaluate_kwargs,
    )
    evaluate(
        problem_type=problem_types,
        output_suffix=f'{output_prefix}/{constraint}_fillna/all',
        framework_nan_fill='constantpredictor_1h8c_gp3_amlb_2023',
        compute_z_score=False,
        **evaluate_kwargs,
    )
    for problem_type in problem_types:
        evaluate(
            problem_type=problem_type,
            output_suffix=f'{output_prefix}/{constraint}/{problem_type}',
            **evaluate_kwargs,
        )
        evaluate(
            problem_type=problem_type,
            output_suffix=f'{output_prefix}/{constraint}_fillna/{problem_type}',
            framework_nan_fill='constantpredictor_1h8c_gp3_amlb_2023',
            compute_z_score=False,
            **evaluate_kwargs,
        )

    from autogluon.common.utils.s3_utils import upload_s3_folder, upload_file
    upload_s3_folder(bucket="autogluon-zeroshot", prefix="autogluon_v1/", folder_to_upload="data/results/output/openml/autogluon_v1/")
    import shutil
    shutil.make_archive("results", 'zip', "data/results/output/openml/autogluon_v1")
    upload_file(file_name="results.zip", bucket="autogluon-zeroshot", prefix="autogluon_v1")
