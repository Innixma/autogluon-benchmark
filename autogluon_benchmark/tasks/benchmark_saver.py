
def get_yaml_str(task_metadata):
    yaml_strs = ['---\n']
    for name, tid in zip(task_metadata['name'], task_metadata['tid']):
        yaml_strs.append(
            f'- name: {name}\n'
            f'  openml_task_id: {tid}\n'
        )
    yaml_str = '\n'.join(yaml_strs)
    return yaml_str
