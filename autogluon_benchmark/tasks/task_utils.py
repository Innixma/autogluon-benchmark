import openml
import time

from openml.exceptions import OpenMLServerException


def get_task(task_id: int):
    task = openml.tasks.get_task(task_id, download_splits=False)
    return task


def get_dataset(task):
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    return X, y


def get_ag_problem_type(task):
    if task.task_type_id.name == 'SUPERVISED_CLASSIFICATION':
        if len(task.class_labels) > 2:
            problem_type = 'multiclass'
        else:
            problem_type = 'binary'
    elif task.task_type_id.name == 'SUPERVISED_REGRESSION':
        problem_type = 'regression'
    else:
        raise AssertionError(f'Unsupported task type: {task.task_type_id.name}')
    return problem_type


def get_task_with_retry(task_id: int, max_delay_exp: int = 8):
    delay_exp = 0
    while True:
        try:
            print(f'Getting task {task_id}')
            task = get_task(task_id=task_id)
            print(f'Got task {task_id}')
            return task
        except OpenMLServerException as e:
            delay = 2 ** delay_exp
            delay_exp += 1
            if delay_exp > max_delay_exp:
                raise ValueError("Unable to get task after 10 retries")
            print(e)
            print(f'Retry in {delay}s...')
            time.sleep(delay)
            continue


def get_task_data(task):
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    return X, y
