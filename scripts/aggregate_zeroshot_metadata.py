import argparse

import pandas as pd

from autogluon.common.loaders import load_pd, load_pkl
from autogluon.common.loaders.load_s3 import list_bucket_prefix_suffix_contains_s3
from autogluon.common.utils import s3_utils
from autogluon.common.savers import save_pkl


def get_s3_paths(path_prefix: str, contains=None, suffix=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix=suffix, contains=contains)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    return paths_full


def aggregate_zeroshot_metadata(path_prefix: str, contains=None, invalid_datasets=None, folds=None):
    zeroshot_suffix = '/zeroshot_metadata.pkl'
    results_suffix = 'scores/results.csv'
    paths_full = get_s3_paths(path_prefix, contains=contains, suffix=zeroshot_suffix)

    if invalid_datasets is None:
        invalid_datasets = set()
    else:
        invalid_datasets = set(invalid_datasets)

    paths_valid = []
    for i, path in enumerate(paths_full):
        dataset_name, fold = path.split('zeroshot/')[1].split('/zeroshot_metadata.pkl')[0].rsplit('/', 1)
        fold = int(fold)
        if folds is not None and fold not in folds:
            continue
        if dataset_name in invalid_datasets:
            continue
        paths_valid.append(path)
    num_paths = len(paths_valid)
    aggregated_pred_proba = {}
    aggregated_ground_truth = {}
    size_bytes_total = 0
    for i, path in enumerate(paths_valid):
        path_to_scores = path.split('/output/', 1)[0] + '/output/' + results_suffix
        scores = load_pd.load(path_to_scores)
        task_name = scores['task'][0]
        id = scores['id'][0]
        fold_in_scores = scores['fold'][0]

        dataset_name, fold = path.split('zeroshot/')[1].split('/zeroshot_metadata.pkl')[0].rsplit('/', 1)
        if task_name != dataset_name:
            print(f'INFO: task name and AWS task name are not the same: {task_name}, {dataset_name}')
        fold = int(fold)
        assert fold == fold_in_scores
        print(f'{i + 1}/{num_paths} | {task_name} | {fold} | {path}')
        try:
            zeroshot_metadata = load_pkl.load(path)
        except Exception:
            continue
        else:
            pass
            import sys
            import pickle
            size_bytes = sys.getsizeof(pickle.dumps(zeroshot_metadata, protocol=4))
            print(f'CUR Size: {round(size_bytes/1e6, 3)} MB')
            size_bytes_total += size_bytes
            print(f'TOT Size: {round(size_bytes_total / 1e6, 3)} MB')

            if task_name not in aggregated_ground_truth:
                aggregated_ground_truth[task_name] = {}
            if fold not in aggregated_ground_truth[task_name]:
                aggregated_ground_truth[task_name][fold] = {}
                for k in [
                    'y_val',
                    'y_test',
                    'eval_metric',
                    'problem_type',
                    'ordered_class_labels',
                    'ordered_class_labels_transformed',
                    'problem_type_transform',
                    'num_classes',
                    'label',
                ]:
                    aggregated_ground_truth[task_name][fold][k] = zeroshot_metadata[k]
                aggregated_ground_truth[task_name][fold]['task'] = task_name
                aggregated_ground_truth[task_name][fold]['id'] = id
            if task_name not in aggregated_pred_proba:
                aggregated_pred_proba[task_name] = {}
            if fold not in aggregated_pred_proba[task_name]:
                aggregated_pred_proba[task_name][fold] = {}
            for k in ['pred_proba_dict_val', 'pred_proba_dict_test']:
                if k not in aggregated_pred_proba[task_name][fold]:
                    aggregated_pred_proba[task_name][fold][k] = {}
                for m, pred_proba in zeroshot_metadata[k].items():
                    aggregated_pred_proba[task_name][fold][k][m] = pred_proba

    return aggregated_pred_proba, aggregated_ground_truth


def aggregate_zeroshot_from_params(s3_bucket, s3_prefix, version_name, constraint, invalid_datasets=None, folds=None):
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
    )

    if len(aggregated_pred_proba) == 0:
        raise AssertionError('Empty Result!')

    aggregated_pred_proba_path = f's3://{s3_bucket}/aggregated/{result_path}zeroshot_pred_proba_{version_name}.pkl'
    aggregated_ground_truth_path = f's3://{s3_bucket}/aggregated/{result_path}zeroshot_gt_{version_name}.pkl'
    print(f'Saving pred_proba output to {aggregated_pred_proba_path}')
    print(f'Saving ground_truth output to {aggregated_ground_truth_path}')

    save_pkl.save(path=aggregated_pred_proba_path, object=aggregated_pred_proba)
    save_pkl.save(path=aggregated_ground_truth_path, object=aggregated_ground_truth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--s3_bucket', type=str, help="Name of S3 bucket that results to aggregate get outputted to",
                        default='automl-benchmark-ag', nargs='?')
    parser.add_argument('--s3_prefix', type=str, help='Prefix for path to results needing aggregation', default='ec2/', nargs='?')
    parser.add_argument('--version_name', type=str, help='Root folder name in EC2 of results', nargs='?')
    parser.add_argument('--constraint', type=str, help='Name of constraint used in benchmark', default='1h8c', nargs='?')
    args = parser.parse_args()

    # constraint = '16h8c'
    # version_name = '2022_10_13_zs'
    # s3_bucket = 'automl-benchmark-ag'
    # s3_prefix = 'ec2/'

    invalid_datasets = [
        'dionis',
        'Airlines_DepDelay_10M',
        'covertype',
        'helena',
        'KDDCup99',
        'sf-police-incidents',
        'Buzzinsocialmedia_Twitter',
        'Higgs',
        'nyc-taxi-green-dec-2016',
        'porto-seguro',
        'volkert',
    ]
    folds = [0]

    aggregate_zeroshot_from_params(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        version_name=args.version_name,
        constraint=args.constraint,
        # s3_bucket=s3_bucket,
        # s3_prefix=s3_prefix,
        # version_name=version_name,
        # constraint=constraint,
        invalid_datasets=invalid_datasets,
        folds=folds,
    )
