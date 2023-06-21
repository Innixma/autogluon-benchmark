from autogluon_benchmark.evaluation.runners.run_evaluation_openml import run


if __name__ == '__main__':
    framework_name_suffix = '_4h8c_gp3_2022_jmlr'
    frameworks_run = [
        'AutoGluon_benchmark',
        'autosklearn',
        # 'autosklearn2',
        'flaml',
        'GAMA_benchmark',
        'H2OAutoML',
        'lightautoml',
        'mljarsupervised_benchmark',
        'TPOT',
        'TunedRandomForest',
        'RandomForest',
        # 'mlr3automl',
    ]
    frameworks_run = [f + framework_name_suffix for f in frameworks_run]

    paths = [
        'amlb/2022_jmlr.csv',  # gjibers et al
    ]

    use_tid_as_dataset_name = False
    problem_types = ['binary', 'multiclass', 'regression']
    treat_folds_as_datasets = False
    # filter_errors = True
    infer_batch_size = None
    run(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=f'2022_amlb_jmlr/1h8c/all',
        problem_type=problem_types,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=True,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
    )
    run(
        paths=paths,
        frameworks_run=frameworks_run,
        output_suffix=f'2022_amlb_jmlr/1h8c_fillna/all',
        framework_nan_fill='constantpredictor_1h8c_gp3_2022_jmlr',
        problem_type=problem_types,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=False,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        compute_z_score=False,
    )
