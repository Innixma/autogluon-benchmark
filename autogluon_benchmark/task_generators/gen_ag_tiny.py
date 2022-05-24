# from ..tasks.task_loader import get_task_dict
# from ..metadata.metadata_loader import load_task_metadata
from autogluon_benchmark.tasks.task_loader import get_task_dict
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata


if __name__ == "__main__":
    task_metadata = load_task_metadata()
    ag_bench = get_task_dict()

    # task_metadata['DataSize'] = task_metadata['NumberOfFeatures'] * task_metadata['NumberOfInstances']
    # task_metadata = task_metadata[task_metadata['DataSize'] <= 1000000]
    # task_metadata = task_metadata[task_metadata['NumberOfFeatures'] <= 100]
    # task_metadata = task_metadata[task_metadata['NumberOfInstances'] >= 1000]

    # Datasets selected based on being able to
    #  1. train fully on AutoGluon-Medium in <60 seconds on a m5.2xlarge.
    #  2. get significant improvement in score when using AutoGluon-Best.
    datasets = [
        359958,  # 'pc4',
        359947,  # 'MIP-2016-regression',
        190392,  # 'madeline',
        359962,  # 'kc1',
        168911,  # 'jasmine',
        359966,  # 'Internet-Advertisements',
        359954,  # 'eucalyptus',
        168757,  # 'credit-g',
        359950,  # 'boston',
        359956,  # 'qsar-biodeg',
        359975,  # 'Satellite',
        359963,  # 'segment',
        359972,  # 'sylvine',
        359934,  # 'tecator',
        146820,  # 'wilt',
    ]

    task_metadata = task_metadata[task_metadata['tid'].isin(datasets)]
    from autogluon_benchmark.tasks.benchmark_saver import get_yaml_str
    yaml_str = get_yaml_str(task_metadata)

    with open("autogluon_benchmark/tasks/ag_tiny.yaml", "w") as text_file:
        text_file.write(yaml_str)
