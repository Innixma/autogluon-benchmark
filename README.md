# AutoGluon AutoML Benchmark Results

Results for AutoGluon Benchmarks

Location of original benchmark results (Non-AutoGluon): https://github.com/openml/automlbenchmark/tree/master/reports

Note: When using the report/results_openml_autogluon_8c1h.csv, the AUC of several problems must be flipped (1-AUC) as automlbenchmark incorrectly scored AutoGluon with flipped results.

Generally, this should be applied to any result with AUC < 0.5.

The version of AutoGluon used for this benchmark is 0.0.3

The code ran in AutoML Benchmark is present in frameworks/autogluon/, to be placed in automlbenchmark/frameworks/autogluon/

AutoGluon: https://github.com/awslabs/autogluon/
