from pathlib import Path

import yaml


def get_task_dict():
    parent_dir = Path(__file__).resolve().parent
    yaml_path = Path.joinpath(parent_dir, 'small.yaml')
    with open(yaml_path, 'r') as stream:
        task_list = yaml.load(stream, Loader=yaml.Loader)
    task_dict = {d['name']: d for d in task_list}
    for task in task_dict.values():
        task.pop('name')
    return task_dict
