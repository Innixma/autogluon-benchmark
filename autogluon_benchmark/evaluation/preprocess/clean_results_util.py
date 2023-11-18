from autogluon_benchmark.evaluation.constants import DATASET, TIME_INFER_S
from autogluon_benchmark.metadata.metadata_loader import load_task_metadata


def clean_data(results_raw, task_metadata_path: str, convert_infer_time_to_per_row: bool = True):
    column_order = list(results_raw.columns)
    task_metadata = load_task_metadata(path=task_metadata_path)
    task_metadata[DATASET] = task_metadata["name"]

    results_raw = results_raw.drop(columns=[DATASET])
    results_raw["tid"] = results_raw["tid"].astype(int)
    pre_unique_tid = set(results_raw["tid"].unique())
    results_raw = results_raw.merge(task_metadata[["NumberOfInstances", DATASET, "tid"]], on="tid")

    post_unique_tid = set(results_raw["tid"].unique())
    if pre_unique_tid != post_unique_tid:
        raise AssertionError(f"Missing TIDs!")

    if convert_infer_time_to_per_row:
        results_raw[TIME_INFER_S] = results_raw[TIME_INFER_S] / results_raw["NumberOfInstances"] * 10
    results_raw = results_raw.drop(columns=["NumberOfInstances"])
    results_raw = results_raw[column_order]
    return results_raw
