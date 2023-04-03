

import openml
import pandas as pd

from autogluon.common.savers import save_pd

from autogluon_benchmark.metadata.metadata_loader import _PATH_TO_DATA


if __name__ == '__main__':
    print('loading complete task list')
    tasks = openml.tasks.list_tasks(output_format="dataframe")
    study_271 = openml.study.get_suite(271)  # Classification 71 datasets
    study_269 = openml.study.get_suite(269)  # Regression 33 datasets
    study_293 = openml.study.get_suite(293)  # Classification 208 datasets
    tasks_271 = tasks[tasks['tid'].isin(set(study_271.tasks))]
    tasks_269 = tasks[tasks['tid'].isin(set(study_269.tasks))]
    tasks_293 = tasks[tasks['tid'].isin(set(study_293.tasks))]
    tasks_amlb = pd.concat([tasks_271, tasks_269, tasks_293], axis=0).reset_index(drop=True)

    print(f'OG Len: {len(tasks_amlb)}')

    task_metadata_og = tasks_amlb.drop_duplicates(subset=['did'])
    task_metadata_og = task_metadata_og.drop_duplicates(subset=['name'])

    # After dedupe, 279 datasets remain
    print(f'Dedupe Len: {len(task_metadata_og)}')
    assert len(task_metadata_og) == 279

    estimation_procedure = '10-fold Crossvalidation'
    estimation_procedure_map = {
        '10-fold Crossvalidation': 1
    }
    estimation_procedure_id = estimation_procedure_map[estimation_procedure]

    print(task_metadata_og)

    task_metadata_valid = task_metadata_og[task_metadata_og['estimation_procedure'] == estimation_procedure]
    tasks_valid = list(task_metadata_valid['tid'])

    task_metadata = task_metadata_og[task_metadata_og['estimation_procedure'] != estimation_procedure]

    datasets_in_task_metadata = set(list(task_metadata['did'].unique()))
    print(len(datasets_in_task_metadata))

    len_task_complete = len(tasks)
    tasks_filtered = tasks[tasks['did'].isin(datasets_in_task_metadata)]
    len_task_filtered = len(tasks_filtered)
    print(f'filtering complete task list to our datasets: {len_task_complete} -> {len_task_filtered}')

    dids = list(task_metadata['did'])
    tasks_valid_additional = []

    missing_dids = []
    for did in dids:
        row_for_did = task_metadata[task_metadata['did'] == did].iloc[0]
        target_feature = row_for_did['target_feature']

        tasks_for_did = tasks_filtered[tasks_filtered['did'] == did]
        tasks_for_did_with_est_proc = tasks_for_did[tasks_for_did['estimation_procedure'] == estimation_procedure]
        tasks_for_did_with_target_feature = tasks_for_did_with_est_proc[tasks_for_did_with_est_proc['target_feature'] == target_feature]
        tasks_for_did_with_evaluation_measures = tasks_for_did_with_target_feature[tasks_for_did_with_target_feature['evaluation_measures'].isna() | (tasks_for_did_with_target_feature['evaluation_measures'] == 'predictive_accuracy')]
        print(f'{did} | {len(tasks_for_did)} | {len(tasks_for_did_with_est_proc)} | {len(tasks_for_did_with_target_feature)} | {len(tasks_for_did_with_evaluation_measures)}')
        if len(tasks_for_did_with_est_proc) != len(tasks_for_did_with_target_feature):
            print('WARNING!!!!!!')
        if len(tasks_for_did_with_evaluation_measures) > 1:
            print(f'\t{len(tasks_for_did_with_evaluation_measures)} | {list(tasks_for_did_with_evaluation_measures["evaluation_measures"])}')
        if len(tasks_for_did_with_evaluation_measures) == 0:
            missing_dids.append(did)
        else:
            tasks_valid_additional.append(tasks_for_did_with_evaluation_measures['tid'].iloc[0])
    print(f'missing {len(missing_dids)}: {missing_dids}')

    # missing_dids = [41160, 41990, 41988, 41986, 41982, 42193, 42343, 42345]

    task_metadata_missing = task_metadata[task_metadata['did'].isin(missing_dids)]

    print(task_metadata_missing)

    tasks_valid_final = tasks_valid + tasks_valid_additional

    print('FINAL TASKS:')
    print(len(tasks_valid_final))
    print(tasks_valid_final)

    task_metadata_final = tasks[tasks['tid'].isin(tasks_valid_final)]

    print(task_metadata_final)

    assert len(task_metadata_final) == len(task_metadata_og)
    save_pd.save(path=f'{_PATH_TO_DATA}/task_metadata_289.csv', df=task_metadata_final)


    # openml.config.apikey = None  # Add API key to add datasets
    # for index, row in task_metadata_missing.iterrows():
    #     print(row['did'])
    #     if row['problem_type'] not in ['binary', 'multiclass']:
    #         raise NotImplementedError
    #     task_type = TaskType.SUPERVISED_CLASSIFICATION
    #     target_name = row['target_feature']
    #     dataset_id = row['did']
    #
    #     print('creating task')
    #     print(f'{task_type} | {dataset_id} | {target_name} | {estimation_procedure_id}')
    #
    #     my_task = openml.tasks.create_task(
    #         task_type=task_type,
    #         dataset_id=dataset_id,
    #         target_name=target_name,
    #         evaluation_measure="predictive_accuracy",
    #         estimation_procedure_id=estimation_procedure_id,
    #     )
    #     my_task.publish()
