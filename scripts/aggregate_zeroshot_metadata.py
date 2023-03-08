import argparse

from autogluon_benchmark.aggregate.zeroshot_metadata import aggregate_zeroshot_from_params


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
    max_size_mb = 10

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
