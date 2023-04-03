

import openml
import pandas as pd

from autogluon.common.savers import save_pd

from autogluon_benchmark.metadata.metadata_loader import load_task_metadata, _PATH_TO_DATA


if __name__ == '__main__':
    task_metadata = load_task_metadata('task_metadata_289.csv', compute_aux_columns=False)
    print('loading complete task_metadata_289')
    print(f'OG Len: {len(task_metadata)}')
    print(task_metadata)
    unique_names = list(task_metadata['name'])
    assert len(task_metadata) == 279
    assert len(unique_names) == 279

    # Datasets with invalid label position used during scoring in AMLB. Likely caused by switch from ARFF to parquet.
    #  These datasets will use the wrong column when trying to score a framework after training.
    banned_datasets_invalid_label_position = [
        # Wrong scores, but doesn't crash
        'compas-two-years',
        'amazon-commerce-reviews',
        'meta_stream_intervals.arff',

        # Crashes during scoring
        'BachChoralHarmony',
        'cjs',
        'irish',
        'JapaneseVowels',
        'KDD98',
        'monks-problems-1',
        'monks-problems-2',
        'monks-problems-3',
        'oil_spill',
        'profb',
        'rl',
        'USPS',
        'vowel',
    ]

    # Datasets whose worst metric_error of any config is <0.001 or whose 20th percentile config error is <0.00025
    banned_datasets_too_easy = [
        'mushroom', 'mv', 'mofn-3-7-10', 'musk', 'skin-segmentation', 'banknote-authentication', 'xd6',
        'analcatdata_supreme', 'kr-vs-kp', 'colleges_aaup', 'arsenic-male-lung', 'stock', 'breast-w', 'wdbc',
        'strikes', 'tic-tac-toe', 'water-treatment', 'arsenic-female-lung', 'arsenic-male-bladder',
    ]

    banned_datasets = banned_datasets_too_easy + banned_datasets_invalid_label_position

    for b in banned_datasets_invalid_label_position:
        assert b in unique_names, f'{b} not in unique_names!'
    task_metadata_final = task_metadata[~task_metadata['name'].isin(banned_datasets)]
    print(task_metadata_final)
    assert len(task_metadata_final) == 244, f'unexpected post-filter length: {len(task_metadata_final)}'

    # FIXME: Remove trivial datasets

    # task_metadata_244
    save_pd.save(path=f'{_PATH_TO_DATA}/task_metadata_244.csv', df=task_metadata_final)
