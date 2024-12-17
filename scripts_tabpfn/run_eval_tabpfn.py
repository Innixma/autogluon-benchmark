import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon_benchmark.evaluation.evaluator import Evaluator


def transform_input_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input df into a valid format to pass into Evaluator

    Once the input data is in the expected format, this transform will no longer be necessary.

    Parameters
    ----------
    input_df

    Returns
    -------

    """
    input_df = input_df.rename({"Unnamed: 0": "dataset"}, axis=1)
    results_tabpfnmix = input_df[["dataset", "tabmix_roc", "tabmix_ce", "tabmix_accuracy"]].copy()
    results_tabpfnmix = results_tabpfnmix.rename({
        "tabmix_roc": "roc_auc",
        "tabmix_ce": "log_loss",
        "tabmix_accuracy": "accuracy",
    }, axis=1)
    results_tabpfnmix["framework"] = "tabpfnmix"

    results_tabforest = input_df[["dataset", "tabforest_roc", "tabforest_ce", "tabforest_accuracy"]].copy()
    results_tabforest = results_tabforest.rename({
        "tabforest_roc": "roc_auc",
        "tabforest_ce": "log_loss",
        "tabforest_accuracy": "accuracy",
    }, axis=1)
    results_tabforest["framework"] = "tabforest"

    results = pd.concat([results_tabpfnmix, results_tabforest], ignore_index=True)

    # FIXME: Hacks since this information isn't present in the original input file
    results["fold"] = 0
    results["problem_type"] = "binary"
    results["metric"] = "roc_auc"
    results["metric_error"] = 1 - results["roc_auc"]
    results["time_train_s"] = 1
    results["time_infer_s"] = 1

    # reorder columns to look nicer
    ordered_columns = [
        "dataset",
        "fold",
        "framework",
        "metric",
        "metric_error",
        "time_train_s",
        "time_infer_s",
        "problem_type",
    ]

    results_columns = ordered_columns + [c for c in results.columns if c not in ordered_columns]
    results = results[results_columns]

    return results


if __name__ == '__main__':
    results_original = load_pd.load("s3://autogluon-zeroshot/tabpfn-results/joined_tabmix_tabforest.csv")
    results = transform_input_df(input_df=results_original)

    # Try uncommenting below to see an example of what running the evaluation code on the TabRepo baselines looks like
    # results = load_pd.load("s3://tabrepo/contexts/2023_11_14/baselines.csv")

    evaluator = Evaluator(worst_fillna=True, frameworks_compare_vs_all=[])
    evaluator_output = evaluator.transform(data=results)

    plotter = evaluator_output.to_plotter(save_dir="output/figures", show=False)
    plotter.plot_all(calibration_elo=1000, BOOTSTRAP_ROUNDS=1000)
