from ..constants import *


def clean_result(result_df, folds_to_keep=None, remove_invalid=True):
    if folds_to_keep is None:
        folds_to_keep = sorted(list(result_df[FOLD].unique()))
    folds_required = len(folds_to_keep)
    result_df = result_df[result_df[FOLD].isin(folds_to_keep)]
    result_df = result_df[result_df[METRIC_ERROR].notnull()]

    if remove_invalid and folds_required > 1:
        results_fold_count_per_run = result_df[[FRAMEWORK, DATASET, FOLD]].groupby([FRAMEWORK, DATASET]).count()
        results_fold_count_per_run_filtered = results_fold_count_per_run[results_fold_count_per_run[FOLD] == folds_required].reset_index()[[FRAMEWORK, DATASET]]
        results_clean_df = result_df.merge(results_fold_count_per_run_filtered, on=[FRAMEWORK, DATASET]).reset_index(drop=True)
    else:
        results_clean_df = result_df.reset_index(drop=True)
    return results_clean_df
