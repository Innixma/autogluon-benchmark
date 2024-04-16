from __future__ import annotations

import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon_benchmark.plotting.plotter import Plotter
from autogluon_benchmark.preprocessing.amlb_preprocessor import AMLBPreprocessor
from autogluon_benchmark.evaluation.evaluator import Evaluator


if __name__ == '__main__':

    s3_prefix = "s3://automl-benchmark-ag/aggregated/release/1.1.0/"
    # task_type = "automm-text"
    task_type = "automm-text-tabular"
    # task_type = "automm-image"

    autogluon_v110 = load_pd.load(f"{s3_prefix}automm-master/{task_type}/results.csv")
    autogluon_v100 = load_pd.load(f"{s3_prefix}automm-v1.0/{task_type}/results_fixed.csv")

    df_raw = pd.concat([
        autogluon_v110,
        autogluon_v100,
    ], ignore_index=True)

    # from autogluon.common.utils.s3_utils import upload_s3_folder
    # upload_s3_folder(bucket="autogluon-zeroshot", prefix="tmpv3/",
    #                  folder_to_upload="../../../amlb_run/2024_04_01")

    df_processed: pd.DataFrame = AMLBPreprocessor().transform(df=df_raw)

    # Renaming to make the plots look nicer
    frameworks_rename = {
        "AutoGluon_benchmark_4h8c": "AutoGluon 0.8 (Best, 4h8c)",
        "AutoGluon_benchmark_1h8c": "AutoGluon 0.8 (Best, 1h8c)",
        "AutoGluon_hq_4h8c": "AutoGluon 0.8 (High, 4h8c)",
        "AutoGluon_hq_1h8c": "AutoGluon 0.8 (High, 1h8c)",

        "flaml_1h8c": "FLAML (2023, 1h8c)",
        "mljarsupervised_benchmark_1h8c": "MLJAR (2023, 1h8c)",
        "H2OAutoML_1h8c": "H2OAutoML (2023, 1h8c)",
        "lightautoml_1h8c": "lightautoml (2023, 1h8c)",
        "autosklearn_1h8c": "autosklearn (2023, 1h8c)",
        "GAMA_benchmark_1h8c": "GAMA (2023, 1h8c)",
        "TPOT_1h8c": "TPOT (2023, 1h8c)",
        "TunedRandomForest_1h8c": "TunedRandomForest (2023, 1h8c)",
        "RandomForest_1h8c": "RandomForest (2023, 1h8c)",
        "constantpredictor_1h8c": "constantpredictor (2023, 1h8c)",

        "flaml_4h8c": "FLAML (2023, 4h8c)",
        "mljarsupervised_benchmark_4h8c": "MLJAR (2023, 4h8c)",
        "H2OAutoML_4h8c": "H2OAutoML (2023, 4h8c)",
        "lightautoml_4h8c": "lightautoml (2023, 4h8c)",
        "autosklearn_4h8c": "autosklearn (2023, 4h8c)",
        "GAMA_benchmark_4h8c": "GAMA (2023, 4h8c)",
        "TPOT_4h8c": "TPOT (2023, 4h8c)",
        "TunedRandomForest_4h8c": "TunedRandomForest (2023, 4h8c)",
        "RandomForest_4h8c": "RandomForest (2023, 4h8c)",
        "constantpredictor_4h8c": "constantpredictor (2023, 4h8c)",

        "AutoGluon_bq_v1_4h8c": "AutoGluon 1.0 (Best, 4h8c)",
        "AutoGluon_gq_v1_4h8c": "AutoGluon 1.0 (Good, 4h8c)",
        "AutoGluon_hq_v1_4h8c": "AutoGluon 1.0 (High, 4h8c)",
        "AutoGluon_hq_v1_il0001_4h8c": "AutoGluon 1.0 (High, 4h8c, infer_limit=0.001)",
        "AutoGluon_hq_v1_il00005_4h8c": "AutoGluon 1.0 (High, 4h8c, infer_limit=0.0005)",
        "AutoGluon_hq_v1_il00001_4h8c": "AutoGluon 1.0 (High, 4h8c, infer_limit=0.0001)",
        "AutoGluon_hq_v1_il000005_4h8c": "AutoGluon 1.0 (High, 4h8c, infer_limit=0.00005)",
        "AutoGluon_benchmark_4h8c_gp3_amlb_2022": "AutoGluon 0.3.1 (Best, 4h8c)",

        "AutoGluon_CatBoost_4h8c": "CatBoost (2023, 4h8c)",
        "AutoGluon_LightGBM_4h8c": "LightGBM (2023, 4h8c)",
        "AutoGluon_XGBoost_4h8c": "XGBoost (2023, 4h8c)",
        "AutoGluon_CatBoost_1h8c": "CatBoost (2023, 1h8c)",
        "AutoGluon_LightGBM_1h8c": "LightGBM (2023, 1h8c)",
        "AutoGluon_XGBoost_1h8c": "XGBoost (2023, 1h8c)",

        "AutoGluon_bq_1h8c": "AutoGluon 1.0 (Best, 1h8c)",
        "AutoGluon_bq_30m8c": "AutoGluon 1.0 (Best, 30m8c)",
        "AutoGluon_bq_10m8c": "AutoGluon 1.0 (Best, 10m8c)",
        "AutoGluon_bq_5m8c": "AutoGluon 1.0 (Best, 5m8c)",

        "AutoGluon_tabular_best_1h8c": "AutoGluon 1.1 Preview (Best, 1h8c)",
        "AutoGluon_best_1h8c": "AutoGluon 1.0 Compare (Best, 1h8c)",
    }

    frameworks = [
        # 'AutoGluon_bq_v1_4h8c',  # AutoGluon 1.0, Best Quality
        # 'AutoGluon_bq_v1_1h8c',  # AutoGluon 1.0, Best Quality
        # 'AutoGluon_benchmark_4h8c',  # AutoGluon 0.8, Best Quality
        'AutoGluon_benchmark_1h8c',  # AutoGluon 0.8, Best Quality

        # 'AutoGluon_hq_v1_4h8c',
        # 'AutoGluon_hq_v1_il00001_4h8c',
        # 'AutoGluon_hq_v1_il00005_4h8c',
        # 'AutoGluon_hq_v1_il0001_4h8c',

        # 'AutoGluon_bq_v1_1h8c',
        # 'AutoGluon_bq_v1_30m8c',
        # 'AutoGluon_bq_v1_10m8c',
        # 'AutoGluon_bq_v1_5m8c',

        'H2OAutoML_1h8c',
        'autosklearn_1h8c',

        'lightautoml_1h8c',
        'GAMA_benchmark_1h8c',
        'flaml_1h8c',
        'mljarsupervised_benchmark_1h8c',
        # 'autosklearn2_4h8c',  # Exclude due to no results for regression tasks
        # 'TPOT_4h8c',  # Exclude due to poor performance and inaccurate inference time
        'TunedRandomForest_1h8c',
        'RandomForest_1h8c',

        "AutoGluon_CatBoost_1h8c",
        "AutoGluon_LightGBM_1h8c",
        "AutoGluon_XGBoost_1h8c",
        # "constantpredictor_4h8c",
    ]
    frameworks += [
        # "AutoGluon_bq_v1_4h8c",
        # "AutoGluon_bq_1.0.0_1h8c",
        # "AutoGluon_bq_1.0.1b20240402_1h8c",
        # "AutoGluon_ray_1h",
        # "AutoGluon_hq_v1_4h8c",
        # "AutoGluon_hq_1.0.0_4h8c",
        # "AutoGluon_hq_1.0.1b20240402_4h8c",
    ]

    frameworks = [
        "AutoGluon_tabular_best_1h8c",
        # "AutoGluon_best_1h8c",
        "AutoGluon_bq_1h8c",
    ] + frameworks

    frameworks = None

    calibration_framework = frameworks_rename["RandomForest_1h8c"]
    calibration_elo = 1000

    evaluator = Evaluator(
        frameworks=frameworks,
        frameworks_rename=frameworks_rename,
        # frameworks_compare_vs_all=["AutoGluon 1.1 Preview (Best, 1h8c)"],
        # frameworks=["AutoGluon_benchmark_1h8c", "AutoGluon_hq_4h8c_LightGBM_r15_BAG_L1_FULL"],
        # folds=[0],
        # framework_fillna="constantpredictor_4h8c",
        # treat_folds_as_datasets=True,
        # use_tid_as_dataset_name=True,
        # task_metadata="task_metadata.csv",
        # verbose=False,
    )

    evaluator_output = evaluator.transform(data=df_processed)

    results_ranked_df = evaluator_output.results_ranked
    results_ranked_fillna_df = evaluator_output.results_ranked

    # plotter: Plotter = evaluator.to_plotter(save_dir="tmpdir")

    plotter = Plotter(
        results_ranked_df=results_ranked_df,
        results_ranked_fillna_df=results_ranked_fillna_df,
        save_dir="tmpdir3",
    )

    plotter.plot_all(
        # calibration_framework=calibration_framework,
        #calibration_elo=calibration_elo,
        BOOTSTRAP_ROUNDS=1000,
    )



"""
|   Rank | Model                              |   Elo | 95% CI   |   Winrate |   Rescaled Acc |   Champ Delta % |
|-------:|:-----------------------------------|------:|:---------|----------:|---------------:|----------------:|
|      1 | AutoGluon 1.1 Preview (Best, 1h8c) |  1750 | +63/-58  |      0.85 |           0.94 |             3.4 |
|      1 | AutoGluon 1.0 (Best, 1h8c)         |  1731 | +60/-62  |      0.84 |           0.94 |             4.3 |
|      3 | AutoGluon 0.8 (Best, 1h8c)         |  1619 | +61/-49  |      0.74 |           0.88 |             8.2 |
|      4 | FLAML (2023, 4h8c)                 |  1507 | +41/-47  |      0.63 |           0.77 |            14.3 |
|      4 | lightautoml (2023, 4h8c)           |  1504 | +52/-47  |      0.63 |           0.79 |            12.8 |
|      4 | H2OAutoML (2023, 4h8c)             |  1472 | +50/-47  |      0.59 |           0.78 |            14.3 |
|      7 | autosklearn (2023, 4h8c)           |  1376 | +49/-44  |      0.5  |           0.69 |            14   |
|      7 | CatBoost (2023, 4h8c)              |  1300 | +49/-39  |      0.42 |           0.6  |            18.8 |
|      9 | TunedRandomForest (2023, 4h8c)     |  1113 | +46/-48  |      0.23 |           0.32 |            25.2 |
|      9 | LightGBM (2023, 4h8c)              |  1112 | +46/-42  |      0.24 |           0.41 |            24.5 |
|      9 | XGBoost (2023, 4h8c)               |  1052 | +35/-46  |      0.18 |           0.38 |            24.8 |
|     12 | RandomForest (2023, 4h8c)          |  1000 | +0/-0    |      0.14 |           0.23 |            27.6 |

|   Rank | Model                              |   Elo | 95% CI   |   Winrate |   Rescaled Acc |   Champ Delta % |
|-------:|:-----------------------------------|------:|:---------|----------:|---------------:|----------------:|
|      1 | AutoGluon 1.1 Preview (Best, 1h8c) |  1632 | +41/-34  |      0.85 |           0.95 |             4.5 |
|      1 | AutoGluon 1.0 (Best, 1h8c)         |  1611 | +38/-36  |      0.84 |           0.95 |             5.2 |
|      3 | AutoGluon 0.8 (Best, 1h8c)         |  1537 | +33/-32  |      0.78 |           0.91 |             8.8 |
|      4 | lightautoml (2023, 1h8c)           |  1366 | +30/-24  |      0.6  |           0.8  |            15.4 |
|      4 | FLAML (2023, 1h8c)                 |  1363 | +33/-29  |      0.59 |           0.8  |            17.1 |
|      4 | H2OAutoML (2023, 1h8c)             |  1326 | +29/-29  |      0.55 |           0.81 |            17.1 |
|      4 | MLJAR (2023, 1h8c)                 |  1312 | +37/-33  |      0.53 |           0.67 |            25.9 |
|      8 | autosklearn (2023, 1h8c)           |  1270 | +35/-26  |      0.49 |           0.75 |            17.4 |
|      8 | AutoGluon_CatBoost_1h8c            |  1233 | +33/-27  |      0.44 |           0.72 |            20.2 |
|     10 | GAMA (2023, 1h8c)                  |  1167 | +30/-29  |      0.36 |           0.59 |            24.7 |
|     11 | AutoGluon_LightGBM_1h8c            |  1091 | +35/-25  |      0.28 |           0.6  |            25.3 |
|     11 | TunedRandomForest (2023, 1h8c)     |  1079 | +30/-30  |      0.26 |           0.51 |            26.6 |
|     11 | AutoGluon_XGBoost_1h8c             |  1052 | +31/-34  |      0.24 |           0.58 |            24   |
|     14 | RandomForest (2023, 1h8c)          |  1000 | +0/-0    |      0.19 |           0.45 |            28   |


|   Rank | Model                              |   Elo | 95% CI   |   Winrate |   Rescaled Acc |   Champ Delta % |
|-------:|:-----------------------------------|------:|:---------|----------:|---------------:|----------------:|
|      1 | AutoGluon 1.1 Preview (Best, 1h8c) |  1633 | +40/-40  |      0.85 |           0.95 |             4.5 |
|      1 | AutoGluon 1.0 (Best, 1h8c)         |  1610 | +31/-38  |      0.84 |           0.95 |             5.2 |
|      3 | AutoGluon 0.8 (Best, 1h8c)         |  1536 | +31/-34  |      0.78 |           0.91 |             8.8 |
|      4 | lightautoml (2023, 1h8c)           |  1364 | +32/-31  |      0.6  |           0.8  |            15.4 |
|      4 | FLAML (2023, 1h8c)                 |  1361 | +32/-34  |      0.59 |           0.8  |            17.1 |
|      4 | H2OAutoML (2023, 1h8c)             |  1322 | +32/-26  |      0.55 |           0.81 |            17.1 |
|      4 | MLJAR (2023, 1h8c)                 |  1312 | +35/-35  |      0.53 |           0.67 |            25.9 |
|      8 | autosklearn (2023, 1h8c)           |  1271 | +28/-36  |      0.49 |           0.75 |            17.4 |
|      8 | CatBoost (2023, 1h8c)              |  1232 | +30/-28  |      0.44 |           0.72 |            20.2 |
|     10 | GAMA (2023, 1h8c)                  |  1170 | +33/-37  |      0.36 |           0.59 |            24.7 |
|     11 | LightGBM (2023, 1h8c)              |  1094 | +29/-30  |      0.28 |           0.6  |            25.3 |
|     11 | TunedRandomForest (2023, 1h8c)     |  1080 | +30/-32  |      0.26 |           0.51 |            26.6 |
|     11 | XGBoost (2023, 1h8c)               |  1054 | +35/-32  |      0.24 |           0.58 |            24   |
|     14 | RandomForest (2023, 1h8c)          |  1000 | +0/-0    |      0.19 |           0.45 |            28   |

                             framework   Winrate    >   <    =  % Loss Reduction  % Loss Reduction (median)  Avg Inf Speed Diff  time_train_s  time_infer_s  loss_rescaled  time_train_s_rescaled  time_infer_s_rescaled       rank  rank=1_count  rank=2_count  rank=3_count  rank>3_count  error_count
0   AutoGluon 1.1 Preview (Best, 1h8c)  0.500000    0   0  104          0.000000                   0.000000            0.000000   3520.856538      0.048170       0.045979             885.436447             844.056602   2.903846            27            34            16            27            0
1           AutoGluon 1.0 (Best, 1h8c)  0.538462   54  46    4          0.279614                   0.028011           -0.035093   3524.744808      0.042365       0.048164             877.636719             837.705779   3.072115            17            35            21            31            0
2           AutoGluon 0.8 (Best, 1h8c)  0.682692   71  33    0          4.983765                   0.598708            0.099833   2951.996731      0.030751       0.085734             662.313987             975.640357   3.913462            15            12            40            37            0
3             lightautoml (2023, 1h8c)  0.836538   87  17    0         12.177237                   4.259097            4.547376   3064.328824      0.233089       0.200184             853.544695             217.602680   6.221154            12             2             4            86            0
4                   FLAML (2023, 1h8c)  0.836538   87  17    0         12.186523                   4.944340          131.704323   3719.515645      0.002272       0.198833             956.891824              20.662591   6.274038             6             4             6            88            0
5               H2OAutoML (2023, 1h8c)  0.875000   91  13    0         11.883103                   5.396372           28.528974   3592.738750      0.002030       0.188835             933.729696              45.349881   6.841346             3             2             3            96            0


|   Rank | Model                              |   Elo | 95% CI   |   Winrate |   Rescaled Acc |   Champ Delta % |
|-------:|:-----------------------------------|------:|:---------|----------:|---------------:|----------------:|
|      1 | AutoGluon 1.1 Preview (Best, 1h8c) |  1633 | +40/-40  |      0.85 |           0.95 |             4.5 |
|      1 | AutoGluon 1.0 (Best, 1h8c)         |  1610 | +31/-38  |      0.84 |           0.95 |             5.2 |
|      3 | AutoGluon 0.8 (Best, 1h8c)         |  1536 | +31/-34  |      0.78 |           0.91 |             8.8 |
|      4 | lightautoml (2023, 1h8c)           |  1364 | +32/-31  |      0.6  |           0.8  |            15.4 |
|      4 | FLAML (2023, 1h8c)                 |  1361 | +32/-34  |      0.59 |           0.8  |            17.1 |
|      4 | H2OAutoML (2023, 1h8c)             |  1322 | +32/-26  |      0.55 |           0.81 |            17.1 |
|      4 | MLJAR (2023, 1h8c)                 |  1312 | +35/-35  |      0.53 |           0.67 |            25.9 |
|      8 | autosklearn (2023, 1h8c)           |  1271 | +28/-36  |      0.49 |           0.75 |            17.4 |
|      8 | CatBoost (2023, 1h8c)              |  1232 | +30/-28  |      0.44 |           0.72 |            20.2 |
|     10 | GAMA (2023, 1h8c)                  |  1170 | +33/-37  |      0.36 |           0.59 |            24.7 |
|     11 | LightGBM (2023, 1h8c)              |  1094 | +29/-30  |      0.28 |           0.6  |            25.3 |
|     11 | TunedRandomForest (2023, 1h8c)     |  1080 | +30/-32  |      0.26 |           0.51 |            26.6 |
|     11 | XGBoost (2023, 1h8c)               |  1054 | +35/-32  |      0.24 |           0.58 |            24   |
|     14 | RandomForest (2023, 1h8c)          |  1000 | +0/-0    |      0.19 |           0.45 |            28   |


"""