from autogluon.bench.eval.evaluation import evaluate_results
from autogluon.bench.eval.evaluation.constants import TIME_INFER_S
from autogluon.bench.eval.evaluation.benchmark_evaluator import BenchmarkEvaluator


if __name__ == '__main__':
    INPUT_PATH = "2023_08_28"

    results_dir = 'data/results/'

    s3_input_dir = 's3://automl-benchmark-ag/aggregated/ec2'

    paths = [
        f'{s3_input_dir}/{INPUT_PATH}/results_preprocessed.csv',
    ]

    problem_types = ['binary', 'multiclass', 'regression']
    # problem_types = ['regression']
    folds = [0]

    benchmark_evaluator = BenchmarkEvaluator(results_dir=results_dir, task_metadata='task_metadata_244.csv')
    results = benchmark_evaluator.load_data(
        paths=paths,
        folds=folds,
        problem_type=problem_types,
    )

    frameworks = list(results['framework'].unique())
    evaluate_results.evaluate(
        results_raw=results,
        frameworks=frameworks,
        columns_to_agg_extra=[
            TIME_INFER_S,
        ],
        frameworks_compare_vs_all=["AutoGluon_bq_1h8c_2023_08_28"],
        output_dir=results_dir,
    )
