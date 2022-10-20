import argparse

import pandas as pd

from autogluon.common.loaders import load_pd, load_pkl
from autogluon.common.loaders.load_s3 import list_bucket_prefix_suffix_s3, list_bucket_prefix_suffix_contains_s3
from autogluon.common.utils import s3_utils
from autogluon.common.savers import save_pd, save_pkl


def get_s3_paths(path_prefix: str, contains=None, suffix=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix, suffix=suffix, contains=contains)
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in objects]
    return paths_full


def aggregate_zeroshot_metadata(path_prefix: str, contains=None, invalid_datasets=None, folds=None):
    paths_full = get_s3_paths(path_prefix, contains=contains, suffix='/zeroshot_metadata.pkl')

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
        dataset_name, fold = path.split('zeroshot/')[1].split('/zeroshot_metadata.pkl')[0].rsplit('/', 1)
        fold = int(fold)
        print(f'{i + 1}/{num_paths} | {dataset_name} | {fold} | {path}')
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

            if dataset_name not in aggregated_ground_truth:
                aggregated_ground_truth[dataset_name] = {}
            if fold not in aggregated_ground_truth[dataset_name]:
                aggregated_ground_truth[dataset_name][fold] = {}
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
                    aggregated_ground_truth[dataset_name][fold][k] = zeroshot_metadata[k]
            if dataset_name not in aggregated_pred_proba:
                aggregated_pred_proba[dataset_name] = {}
            if fold not in aggregated_pred_proba[dataset_name]:
                aggregated_pred_proba[dataset_name][fold] = {}
            for k in ['pred_proba_dict_val', 'pred_proba_dict_test']:
                if k not in aggregated_pred_proba[dataset_name][fold]:
                    aggregated_pred_proba[dataset_name][fold][k] = {}
                for m, pred_proba in zeroshot_metadata[k].items():
                    aggregated_pred_proba[dataset_name][fold][k][m] = pred_proba

    return aggregated_pred_proba, aggregated_ground_truth


def aggregate_zeroshot_from_params(s3_bucket, s3_prefix, version_name, constraint, invalid_datasets=None, folds=None):
    contains = f'.{constraint}.'
    result_path = f'{s3_prefix}{version_name}/'

    aggregated_pred_proba, aggregated_ground_truth = aggregate_zeroshot_metadata(
        path_prefix=f's3://{s3_bucket}/{result_path}',
        contains=contains,
        invalid_datasets=invalid_datasets,
        folds=folds,
    )

    save_pkl.save(path=f's3://{s3_bucket}/aggregated/{result_path}zeroshot_pred_proba_{version_name}.pkl', object=aggregated_pred_proba)
    save_pkl.save(path=f's3://{s3_bucket}/aggregated/{result_path}zeroshot_gt_{version_name}.pkl', object=aggregated_ground_truth)

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
        invalid_datasets=invalid_datasets,
        folds=folds,
    )
