# WARNING: This repo is OUTDATED! Please go to https://github.com/Innixma/autogluon-benchmarking for the full results.








# AutoGluon AutoML Benchmark Results

Results for AutoGluon Benchmarks (see report/results.csv and report/results_openml_autogluon_8c1h.csv)

Location of original benchmark results (Non-AutoGluon): https://github.com/openml/automlbenchmark/tree/master/reports

Note: When using the report/results_openml_autogluon_8c1h.csv, the AUC of several problems must be flipped (1-AUC) as automlbenchmark incorrectly scored AutoGluon with flipped results.

Generally, this should be applied to any result with AUC < 0.5.

## Reproducing Result

To reproduce the results, the benchmark should be ran on an m5.2xlarge EC2 Instance.

The version of AutoGluon used for this benchmark is 0.0.3

This requires building both AutoGluon and automlbenchmark from source.
- automlbenchmark commit: 2f3bb4a6637ea8875abbf7c06f8df649b7f5e2b0
- autogluon commit: 360fb249d55fd90ae4139a234080cc4f44e4820c

The code updates required to run AutoML Benchmark with AutoGluon are present in the provided autogluon and automlbenchmark folders.

These code changes must be placed in their respective source locations.
- AutoGluon: https://github.com/awslabs/autogluon/
- automlbenchmark: https://github.com/openml/automlbenchmark

After updating the files

In terminal:
~~~
pip install --upgrade pip
pip install --upgrade mxnet
cd ~
cd $AUTOGLUON_ROOT
python setup.py develop
cd ~
pip install -r $AUTOMLBENCHMARK_ROOT/requirements.txt
pip install scikit-learn==0.20.4
python $AUTOMLBENCHMARK_ROOT/runbenchmark.py autogluon medium-8c1h
python $AUTOMLBENCHMARK_ROOT/runbenchmark.py autogluon small-8c1h
~~~

Note that this will take a very long time on a single machine (350+ hrs). If you want to truly reproduce the entire benchmark, please contact the AutoGluon team for assistance to setup distributed benchmarking.
