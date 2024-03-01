from pathlib import Path

import pandas as pd

from autogluon.bench.eval.scripts.run_evaluation_openml import evaluate
from autogluon.common.loaders import load_pd
from autogluon_benchmark.plotting.plotter import Plotter
from autogluon_benchmark.preprocessing.amlb_preprocessor import AMLBPreprocessor


"""
| Method                                | AG Winrate   | AG Loss Improvement   |   Rescaled Loss |   Rank | Champion   |
|:--------------------------------------|:-------------|:----------------------|----------------:|-------:|:-----------|
| AutoGluon with Portfolio (Best, 4h8c) | 50%          | 0.0%                  |            0.04 |   2.33 | 42%        |
| AutoGluon (Best, 4h8c)                | 67%          | 2.8%                  |            0.06 |   2.87 | 19%        |
| lightautoml (2023, 4h8c)              | 83%          | 11.7%                 |            0.22 |   5.54 | 12%        |
| H2OAutoML (2023, 4h8c)                | 92%          | 10.3%                 |            0.18 |   5.75 | 1%         |
| FLAML (2023, 4h8c)                    | 87%          | 16.3%                 |            0.24 |   6.08 | 5%         |
| MLJAR (2023, 4h8c)                    | 81%          | 22.5%                 |            0.34 |   6.3  | 6%         |
| autosklearn (2023, 4h8c)              | 89%          | 11.9%                 |            0.24 |   6.83 | 4%         |
| GAMA (2023, 4h8c)                     | 86%          | 15.5%                 |            0.3  |   6.95 | 4%         |
| CatBoost (2023, 4h8c)                 | 94%          | 18.1%                 |            0.29 |   7.59 | 2%         |
| LightGBM (2023, 4h8c)                 | 98%          | 23.6%                 |            0.42 |   9.84 | 0%         |
| TunedRandomForest (2023, 4h8c)        | 94%          | 22.9%                 |            0.51 |   9.87 | 2%         |
| XGBoost (2023, 4h8c)                  | 98%          | 21.0%                 |            0.45 |  10.24 | 1%         |
| RandomForest (2023, 4h8c)             | 97%          | 25.0%                 |            0.58 |  10.82 | 0%         |
"""
if __name__ == '__main__':
    constraint = "4h8c"
    figure_save_dir = "tabrepo_figures"
    show = True
    BOOTSTRAP_ROUNDS = 1000  # Reduce this for a faster execution. Use 1000 for the final plot.

    path_prefix = str(Path(__file__).parent)
    amlb_2023_raw = load_pd.load(f"{path_prefix}/amlb_2023.csv")
    autogluon_v1_raw = load_pd.load(f"{path_prefix}/autogluon_v1_ablation.parquet")

    framework_name_suffix = f'_{constraint}_gp3_amlb_2023'
    frameworks_run_amlb = [
        # 'AutoGluon_benchmark',
        # 'AutoGluon_hq',
        'autosklearn',
        'flaml',
        'H2OAutoML',
        'lightautoml',
        'GAMA_benchmark',
        'mljarsupervised_benchmark',
        # 'mljarsupervised_perform',
        # 'TPOT',
        'TunedRandomForest',
        'RandomForest',
        # # 'NaiveAutoML',
        # 'constantpredictor',
    ]
    frameworks_run = [f + framework_name_suffix for f in frameworks_run_amlb]

    frameworks_run = [
        "AutoGluon_bq_4h8c_2024_02_22",
        "AutoGluon_bq_noportfolio_4h8c_2024_02_22",
        "AutoGluon_XGBoost_4h8c_2024_02_22",
        "AutoGluon_CatBoost_4h8c_2024_02_22",
        "AutoGluon_LightGBM_4h8c_2024_02_22",
        # "AutoGluon_bq_nodynamic_4h8c_2024_02_22",
        # "AutoGluon_bq_noportfolio_nodynamic_4h8c_2024_02_22",
    ] + frameworks_run

    problem_types = [
        'binary',
        'multiclass',
        'regression',
    ]
    use_tid_as_dataset_name = False
    treat_folds_as_datasets = False
    task_metadata = "task_metadata.csv"
    infer_batch_size = None

    banned_datasets = None

    frameworks_rename = {
        "AutoGluon_benchmark_4h8c_gp3_amlb_2023": "AutoGluon 0.8 (Best, 4h8c)",
        "AutoGluon_benchmark_1h8c_gp3_amlb_2023": "AutoGluon 0.8 (Best, 1h8c)",
        "AutoGluon_hq_4h8c_gp3_amlb_2023": "AutoGluon 0.8 (High, 4h8c)",
        "AutoGluon_hq_1h8c_gp3_amlb_2023": "AutoGluon 0.8 (High, 1h8c)",

        "flaml_1h8c_gp3_amlb_2023": "FLAML (2023, 1h8c)",
        "mljarsupervised_benchmark_1h8c_gp3_amlb_2023": "MLJAR (2023, 1h8c)",
        "H2OAutoML_1h8c_gp3_amlb_2023": "H2OAutoML (2023, 1h8c)",
        "lightautoml_1h8c_gp3_amlb_2023": "lightautoml (2023, 1h8c)",
        "autosklearn_1h8c_gp3_amlb_2023": "autosklearn (2023, 1h8c)",
        "GAMA_benchmark_1h8c_gp3_amlb_2023": "GAMA (2023, 1h8c)",
        "TPOT_1h8c_gp3_amlb_2023": "TPOT (2023, 1h8c)",
        "TunedRandomForest_1h8c_gp3_amlb_2023": "TunedRandomForest (2023, 1h8c)",
        "RandomForest_1h8c_gp3_amlb_2023": "RandomForest (2023, 1h8c)",
        "constantpredictor_1h8c_gp3_amlb_2023": "constantpredictor (2023, 1h8c)",

        "flaml_4h8c_gp3_amlb_2023": "FLAML (2023, 4h8c)",
        "mljarsupervised_benchmark_4h8c_gp3_amlb_2023": "MLJAR (2023, 4h8c)",
        "H2OAutoML_4h8c_gp3_amlb_2023": "H2OAutoML (2023, 4h8c)",
        "lightautoml_4h8c_gp3_amlb_2023": "lightautoml (2023, 4h8c)",
        "autosklearn_4h8c_gp3_amlb_2023": "autosklearn (2023, 4h8c)",
        "GAMA_benchmark_4h8c_gp3_amlb_2023": "GAMA (2023, 4h8c)",
        "TPOT_4h8c_gp3_amlb_2023": "TPOT (2023, 4h8c)",
        "TunedRandomForest_4h8c_gp3_amlb_2023": "TunedRandomForest (2023, 4h8c)",
        "RandomForest_4h8c_gp3_amlb_2023": "RandomForest (2023, 4h8c)",
        "constantpredictor_4h8c_gp3_amlb_2023": "constantpredictor (2023, 4h8c)",

        "AutoGluon_bq_v1_4h8c_2023_11_26": "AutoGluon 1.0 (Best, 4h8c)",
        "AutoGluon_gq_v1_4h8c_2023_11_26": "AutoGluon 1.0 (Good, 4h8c)",
        "AutoGluon_hq_v1_4h8c_2023_11_26": "AutoGluon 1.0 (High, 4h8c)",

        "AutoGluon_CatBoost_4h8c_2024_02_22": "CatBoost (2023, 4h8c)",
        "AutoGluon_LightGBM_4h8c_2024_02_22": "LightGBM (2023, 4h8c)",
        "AutoGluon_XGBoost_4h8c_2024_02_22": "XGBoost (2023, 4h8c)",

        "AutoGluon_bq_4h8c_2024_02_22": "AutoGluon with Portfolio (Best, 4h8c)",
        "AutoGluon_bq_noportfolio_4h8c_2024_02_22": "AutoGluon (Best, 4h8c)",
    }

    results_dir = f"{str(Path(__file__).parent / 'data' / 'results')}"
    output_prefix = "tabrepo_v1"

    df_processed_autogluon_v1: pd.DataFrame = AMLBPreprocessor(framework_suffix="2024_02_22").transform(df=autogluon_v1_raw)
    df_processed_amlb_2023: pd.DataFrame = AMLBPreprocessor(framework_suffix="amlb_2023").transform(df=amlb_2023_raw)

    df_processed = pd.concat([
        df_processed_amlb_2023,
        df_processed_autogluon_v1,
    ], ignore_index=True)

    evaluate_kwargs = dict(
        paths=df_processed,
        results_dir=results_dir,
        frameworks_run=frameworks_run,
        treat_folds_as_datasets=treat_folds_as_datasets,
        infer_batch_size=infer_batch_size,
        filter_errors=False,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        task_metadata=task_metadata,
        banned_datasets=banned_datasets,
        frameworks_rename=frameworks_rename,
    )

    _, results_ranked_df, _, _, _ = evaluate(
        problem_type=problem_types,
        output_suffix=f'{output_prefix}/{constraint}/all',
        **evaluate_kwargs,
    )
    _, results_ranked_fillna_df, _, _, _ = evaluate(
        problem_type=problem_types,
        output_suffix=f'{output_prefix}/{constraint}_fillna/all',
        framework_nan_fill='constantpredictor_1h8c_gp3_amlb_2023',
        compute_z_score=False,
        **evaluate_kwargs,
    )

    plotter = Plotter(
        results_ranked_fillna_df=results_ranked_fillna_df,
        results_ranked_df=results_ranked_df,
        save_dir=figure_save_dir,
        show=show,
    )

    plotter.plot_all(
        calibration_framework="RandomForest (2023, 4h8c)",
        calibration_elo=1000,
        BOOTSTRAP_ROUNDS=BOOTSTRAP_ROUNDS,  # Reduce this to lower values for a faster execution. Use 1000 for the final plot.
    )
