import pandas as pd

from autogluon.common.loaders import load_pd

from autogluon_benchmark.plotting.plotter import Plotter


def plot_results(
    results_ranked_fillna_df: pd.DataFrame,
    results_ranked_df: pd.DataFrame,
    save_dir: str,
    show: bool = True,
):
    plotter = Plotter(
        results_ranked_fillna_df=results_ranked_fillna_df,
        results_ranked_df=results_ranked_df,
        save_dir=save_dir,
        show=show,
    )

    plotter.plot_all(
        calibration_framework="RandomForest (2023, 4h8c)",
        calibration_elo=800,
    )


# TODO: Fix the input so that everything is in a function and it can be called at the end of `run_eval_autogluon_v1.py`
if __name__ == '__main__':
    # results_dir = "s3://autogluon-zeroshot/autogluon_v1/"  # s3 path
    results_dir = 'data/results/output/openml/autogluon_v1/'  # local path
    results_dir_input = results_dir
    plotter_root_dir = "test_out3/"

    run_path_prefix = f'4h8c/all/'
    run_path_prefix_fillna = f'4h8c_fillna/all/'
    results_ranked_df_full = load_pd.load(f'{results_dir_input}{run_path_prefix}results_ranked_by_dataset_valid.csv')
    results_ranked_fillna_df_full = load_pd.load(f'{results_dir_input}{run_path_prefix_fillna}results_ranked_by_dataset_valid.csv')

    results_ranked_df_full['dataset'] = results_ranked_df_full['dataset'].astype(str)
    results_ranked_fillna_df_full['dataset'] = results_ranked_fillna_df_full['dataset'].astype(str)

    for problem_type in [
        "all",
        # "binary",
        # "multiclass",
        # "regression",
    ]:
        results_ranked_df = results_ranked_df_full.copy()
        results_ranked_fillna_df = results_ranked_fillna_df_full.copy()

        if problem_type != "all":
            results_ranked_df = results_ranked_df[results_ranked_df["problem_type"] == problem_type]
            results_ranked_fillna_df = results_ranked_fillna_df[results_ranked_fillna_df["problem_type"] == problem_type]

        plotter = Plotter(
            results_ranked_fillna_df=results_ranked_fillna_df,
            results_ranked_df=results_ranked_df,
            save_dir=f"{plotter_root_dir}{problem_type}/",
            show=True,
        )

        plotter.plot_boxplot_rescaled_accuracy()
        plotter.plot_boxplot_champion_loss_delta()
        plotter.plot_boxplot_rank()
        plotter.plot_boxplot_time_train()
        plotter.plot_boxplot_samples_per_second()
        plotter.plot_pareto_time_infer()
        plotter.plot_pareto_time_train()
        plotter.plot_critical_difference()
        plotter.plot_elo_ratings(
            calibration_framework="RandomForest (2023, 4h8c)",
            calibration_elo=800,
        )

    # from autogluon.common.utils.s3_utils import upload_s3_folder, upload_file
    # upload_s3_folder(bucket="autogluon-zeroshot", prefix="autogluon_v1/", folder_to_upload=plotter_root_dir)
    # import shutil
    # shutil.make_archive("results", 'zip', plotter_root_dir)
    # upload_file(file_name="results.zip", bucket="autogluon-zeroshot", prefix="autogluon_v1")
