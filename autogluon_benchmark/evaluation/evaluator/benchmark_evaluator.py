import warnings

import pandas as pd

from autogluon.common.utils.s3_utils import is_s3_url

from ..constants import TIME_INFER_S, METRIC_ERROR
from ..preprocess.preprocess_utils import fill_missing_results_with_default, convert_folds_into_separate_datasets
from ...metadata.metadata_loader import load_task_metadata


class BenchmarkEvaluator:
    def __init__(self,
                 results_dir='data/results/',
                 output_suffix='ag_full_v5/1h8c',
                 use_tid_as_dataset_name: bool = False,
                 filter_errors: bool = False,
                 framework_nan_fill: str = None,
                 ):
        self.results_dir = results_dir
        self.results_dir_input = results_dir + 'input/prepared/openml/'
        self.results_dir_output = results_dir + f'output/openml/{output_suffix}/'
        self._use_tid_as_dataset_name = use_tid_as_dataset_name
        self._filter_errors = filter_errors
        if self._filter_errors:
            framework_nan_fill = None
        self._framework_nan_fill = framework_nan_fill

    def _load_results(self, paths: list) -> pd.DataFrame:
        paths = [path if is_s3_url(path) else self.results_dir_input + path for path in paths]
        results_raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=True)
        results_raw = results_raw.drop_duplicates(subset=['framework', 'dataset', 'fold'])
        self._check_results_valid(results_raw=results_raw)
        return results_raw

    def _check_results_valid(self, results_raw: pd.DataFrame):
        if results_raw[METRIC_ERROR].min() < 0:
            raise AssertionError(f'METRIC_ERROR cannot be negative! There may be a bug. Found min value: {results_raw[METRIC_ERROR].min()}')

    def load_data(self,
                  paths: list,
                  frameworks: list = None,
                  folds: list = None,
                  clean_data: bool = False,
                  problem_type=None,
                  banned_datasets: list = None,
                  infer_batch_size: int = None,
                  treat_folds_as_datasets: bool = False,
                  ) -> pd.DataFrame:
        results_raw = self._load_results(paths=paths)
        if folds is not None:
            results_raw = results_raw[results_raw['fold'].isin(folds)]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            results_raw['time_infer_s'][results_raw['time_infer_s'] == 0] = 0.001
        if clean_data:
            # FIXME: This doesn't work on new tasks due to not comprehensive metadata
            results_raw = self._clean_data(results_raw=results_raw)
        if problem_type is not None:
            if isinstance(problem_type, list):
                results_raw = results_raw[results_raw['problem_type'].isin(problem_type)]
            else:
                results_raw = results_raw[results_raw['problem_type'] == problem_type]
            print(f'Filtering to the following problem_type: {problem_type}')
        if banned_datasets is not None:
            results_raw = results_raw[~results_raw['dataset'].isin(banned_datasets)]
        if self._use_tid_as_dataset_name:
            results_raw['dataset'] = results_raw['tid'].astype(int).astype(str)
            if banned_datasets is not None:
                results_raw = results_raw[~results_raw['dataset'].isin(banned_datasets)]
        if infer_batch_size is not None:
            results_raw = self._update_infer_batch_size(results_raw=results_raw, infer_batch_size=infer_batch_size)
        if self._framework_nan_fill is not None:
            results_raw = fill_missing_results_with_default(framework_nan_fill=self._framework_nan_fill, frameworks_to_fill=frameworks, results_raw=results_raw)
        if frameworks is not None:
            results_raw = self._filter_frameworks(results_raw=results_raw, frameworks=frameworks)
        if treat_folds_as_datasets:
            results_raw = convert_folds_into_separate_datasets(results_raw=results_raw)
            folds = [0]
        if self._filter_errors:
            results_raw = self.filter_errors(results_raw=results_raw, folds=folds)
        if frameworks is not None:
            frameworks_present = list(results_raw['framework'].unique())
            assert set(frameworks) == set(frameworks_present)
        return results_raw

    def _clean_data(self, results_raw):
        task_metadata = load_task_metadata()
        task_metadata['dataset'] = task_metadata['name']
        # FIXME: TEMP
        results_raw = results_raw.drop(columns=['tid'])
        results_raw['dataset'] = results_raw['dataset'].map({'numerai28_6': 'numerai28.6'}).fillna(results_raw['dataset'])
        results_raw = results_raw.merge(task_metadata[['NumberOfInstances', 'dataset', 'tid']], on='dataset')
        # FIXME: TEMP
        results_raw[TIME_INFER_S] = results_raw[TIME_INFER_S] / results_raw['NumberOfInstances'] * 10
        return results_raw

    def _update_infer_batch_size(self, results_raw: pd.DataFrame, infer_batch_size: int):
        # Update infer time
        if f'pred_time_test_with_transform_batch_size_{infer_batch_size}' in results_raw.columns:
            results_raw['time_infer_s'] = results_raw[f'pred_time_test_with_transform_batch_size_{infer_batch_size}'].fillna(results_raw['time_infer_s'])
        if f'pred_time_test_with_transform_{infer_batch_size}' in results_raw.columns:
            results_raw['time_infer_s'] = results_raw[f'pred_time_test_with_transform_{infer_batch_size}'].fillna(results_raw['time_infer_s'])
        return results_raw

    def filter_errors(self, results_raw: pd.DataFrame, folds, frameworks: list = None):
        """
        For each framework in frameworks, ensure that only datasets without failures from this framework are kept.
        """
        # FIXME: Ensure correct folds, not just count
        if frameworks is None:
            frameworks = list(results_raw['framework'].unique())
        for f in frameworks:
            datasets_keep = results_raw[results_raw['framework'] == f]['dataset'].value_counts()
            datasets_keep = list(datasets_keep[datasets_keep == len(folds)].index)
            results_raw = results_raw[results_raw['dataset'].isin(datasets_keep)]
        return results_raw

    def _filter_frameworks(self, results_raw: pd.DataFrame, frameworks: list):
        return results_raw[results_raw['framework'].isin(frameworks)]
