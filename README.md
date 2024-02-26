Code for benchmarking AutoGluon.

## Development Installation

To get started, run the following commands:

```
# Do this if you are locally developing AutoGluon to avoid installing it from pip:
git clone https://github.com/autogluon/autogluon
cd autogluon
./full_install.sh
cd ..
```

Install AutoGluon Bench if running evaluation code.
```
git clone https://github.com/autogluon/autogluon-bench.git
cd autogluon-bench
pip install -e .
cd ..
```

Install AutoGluon Benchmark
```
# Install autogluon-benchmark
git clone https://github.com/Innixma/autogluon-benchmark.git
cd autogluon-benchmark
pip install -e .
```

## Full AutoMLBenchmark

To run AutoMLBenchmark, see instructions in:

`examples/automlbenchmark/README_automlbenchmark.md`

Please note that these instructions are quite technical.

## Local Testing

To benchmark individual OpenML datasets, you can check out the examples in `examples/train_***`

The scripts are fairly primitive. They mimic what AutoMLBenchmark does contained to a single dataset.

## Generating Task Metadata

To generate task metadata files, refer to autogluon_benchmark/data/metadata/README.md
