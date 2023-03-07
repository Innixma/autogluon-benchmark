import argparse

from autogluon.common.savers import save_pkl

from autogluon_benchmark import OutputSuiteContext


def aggregate_zeroshot_metadata(path_prefix: str, contains=None, invalid_datasets=None, folds=None, max_size_mb=10):
    output_suite_context = OutputSuiteContext(
        path=path_prefix,
        contains=contains,
        mode='ray',
    )
    if invalid_datasets is None:
        invalid_datasets = set()
    else:
        invalid_datasets = set(invalid_datasets)

    output_suite_context.filter_failures()
    def _validate(output_context):
        is_valid = True
        result_df = output_context.load_results()
        fold = result_df.iloc[0]['fold']
        task_name = result_df.iloc[0]['task']
        fold = int(fold)
        if folds is not None and fold not in folds:
            is_valid = False
        if task_name in invalid_datasets:
            print(f'INVALID TASK: {task_name}')
            is_valid = False
        return is_valid

    is_valid_lst = output_suite_context._loop_func(func=_validate, input_list=output_suite_context.output_contexts)
    output_suite_context.filter(is_valid_lst)

    zeroshot_metadata_list = output_suite_context.load_zeroshot_metadata(max_size_mb=max_size_mb, allow_exception=True)

    output_suite_context.filter(filter_lst=[zsm is not None for zsm in zeroshot_metadata_list])

    import sys
    import pickle
    size_bytes_total = 0
    len_total = len(zeroshot_metadata_list)
    zeroshot_metadata_list = [z for z in zeroshot_metadata_list if z is not None]
    len_valid = len(zeroshot_metadata_list)
    for i, zsm in enumerate(zeroshot_metadata_list):
        size_bytes = sys.getsizeof(pickle.dumps(zsm, protocol=4))
        size_bytes_total += size_bytes
        print(f'TOT Size: {round(size_bytes_total / 1e6, 3)} MB | CUR Size: {round(size_bytes / 1e6, 3)} MB')
    print(f'{len_valid}/{len_total} Valid Results!')

    results_lst = output_suite_context.load_results()

    aggregated_pred_proba, aggregated_ground_truth = output_suite_context.construct_zs_dict(results_lst=results_lst, zeroshot_metadata_list=zeroshot_metadata_list)

    # output_suite_context._filter_zs(aggregated_pred_proba=aggregated_pred_proba,
    #                                 aggregated_ground_truth=aggregated_ground_truth,
    #                                 require_all_models=True)

    # from autogluon_zeroshot.simulation.tabular_predictions import TabularPicklePredictions
    # tpp = TabularPicklePredictions(pred_dict=aggregated_pred_proba)

    return aggregated_pred_proba, aggregated_ground_truth


def aggregate_zeroshot_from_params(s3_bucket,
                                   s3_prefix,
                                   version_name,
                                   constraint,
                                   invalid_datasets=None,
                                   folds=None,
                                   max_size_mb=10):
    assert version_name is not None
    assert s3_bucket is not None
    assert s3_prefix is not None
    assert constraint is not None
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'

    aggregated_pred_proba, aggregated_ground_truth = aggregate_zeroshot_metadata(
        path_prefix=f's3://{s3_bucket}/{result_path}',
        contains=contains,
        invalid_datasets=invalid_datasets,
        folds=folds,
        max_size_mb=max_size_mb,
    )

    if len(aggregated_pred_proba) == 0:
        raise AssertionError('Empty Result!')

    aggregated_pred_proba_path = f's3://{s3_bucket}/aggregated/{result_path}zeroshot_pred_proba_{version_name}_mini.pkl'
    aggregated_ground_truth_path = f's3://{s3_bucket}/aggregated/{result_path}zeroshot_gt_{version_name}_mini.pkl'
    print(f'Saving pred_proba output to {aggregated_pred_proba_path}')
    print(f'Saving ground_truth output to {aggregated_ground_truth_path}')

    save_pkl.save(path=aggregated_pred_proba_path, object=aggregated_pred_proba)
    save_pkl.save(path=aggregated_ground_truth_path, object=aggregated_ground_truth)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
    #                     default='automl-benchmark-ag', nargs='?')
    # parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    # parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    # parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default='1h8c', nargs='?')
    # args = parser.parse_args()

    constraint = '24h64c'
    version_name = '2023_02_27_zs'
    s3_bucket = 'automl-benchmark-ag'
    s3_prefix = 'ec2/'
    max_size_mb = 100

    # invalid_datasets = [
    #     'dionis',
    #     'Airlines_DepDelay_10M',
    #     'covertype',
    #     'helena',
    #     'KDDCup99',
    #     'sf-police-incidents',
    #     'Buzzinsocialmedia_Twitter',
    #     'Higgs',
    #     'nyc-taxi-green-dec-2016',
    #     'porto-seguro',
    #     'volkert',
    # ]
    invalid_datasets = []
    # folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    folds = [0]

    aggregate_zeroshot_from_params(
        # s3_bucket=args.s3_bucket,
        # s3_prefix=args.s3_prefix,
        # version_name=args.version_name,
        # constraint=args.constraint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        version_name=version_name,
        constraint=constraint,
        invalid_datasets=invalid_datasets,
        max_size_mb=max_size_mb,
        folds=folds,
    )
