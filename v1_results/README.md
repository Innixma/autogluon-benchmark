Files in this directory contain results for AutoGluon 1.0

## Raw Result Files

1. `autogluon_v1.csv`: Contains the raw AutoML Benchmark results for AutoGluon 1.0.
2. `amlb_2023.csv`: Contains the official raw AutoML Benchmark results for the [AutoML Benchmark 2023](https://arxiv.org/abs/2207.12560).

We are working on cleaning up our plot and rank generation code so we can provide a directly runnable way to reproduce our figures from the raw data. Simulatenously, we are working with the AutoML Benchmark authors to make our results available on [the official AutoML Benchmark website](https://openml.github.io/automlbenchmark/index.html).

## Notes on experiments

1. AutoGluon 1.0 results were obtained on a pre-release version of AutoGluon 1.0 on November 26th, but should be roughly equivalent to the state of the released version on November 29th.
2. AutoGluon 1.0 High Quality results go slightly over the time limit provided by ~10% on 25% of the datasets. This is due to a preset name mismatch in the benchmark that normally reduces the time given to `high_quality` by 10% to better adhere to time limits as mentioned in the AutoML Benchmark 2023 paper. We don't believe that the 10% time difference changes the results meaningfully however.
3. LightGBM, XGBoost, and CatBoost results were obtained by running AutoGluon on `medium_quality` setting and obtaining the individual model results for LightGBM, XGBoost, and CatBoost trained via AutoGluon. These results are stronger than the native defaults of these models due to AutoGluon's optimizations. Performing extensive HPO on these models can lead to some improvement, but we note that HPO does not improve performance much for these models, as covered in detail in the [TabRepo paper](https://arxiv.org/abs/2311.02971v1) Table 1. We hope to extend our comparison to HPO tuned variants of these models soon.
4. All other results are obtained directly from the raw results file of the [AutoML Benchmark 2023 paper](https://arxiv.org/abs/2207.12560) published on November 16th 2023.
5. All experiments were run on m5.2xlarge EC2 instances.
6. For calculating inference speed, we use the method in the AutoML Benchmark 2021 paper rather than the 2023 paper because we had not tested the newly implemented inference speed measurement logic. We have discussed with the AMLB authors and agreed that this should not change the take-aways from the results, as both methods are fairly consistent with eachother.
