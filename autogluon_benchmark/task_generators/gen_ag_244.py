
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata
from autogluon_benchmark.tasks.benchmark_saver import get_yaml_str

if __name__ == "__main__":
    task_metadata = load_task_metadata('task_metadata_244.csv')

    yaml_str = get_yaml_str(task_metadata)

    with open("../tasks/ag_244.yaml", "w") as text_file:
        text_file.write(yaml_str)
