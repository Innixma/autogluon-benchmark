from __future__ import annotations

from typing import Dict, List

import pandas as pd

from autogluon.bench.eval.scripts.run_evaluation_openml import evaluate
from autogluon_benchmark.plotting.plotter import Plotter


# FIXME: Differentiate between `time_infer_s` and `time_infer_s_per_sample`
# FIXME: Update to `transform` API?
class Evaluator:
    def __init__(
        self,
        df_processed: pd.DataFrame,
        frameworks: List[str] | None = None,
        framework_fillna: str | None = None,
        frameworks_rename: Dict[str, str] | None = None,
        task_metadata: pd.DataFrame | None = None,
        clean_data: bool | str = "auto",
        folds: List[int] | None = None,
    ):
        self.frameworks = frameworks
        self.framework_fillna = framework_fillna
        self.frameworks_rename = frameworks_rename
        self.task_metadata = task_metadata
        self.folds = folds
        self.df_processed = df_processed
        if clean_data == "auto":
            clean_data = self.task_metadata is not None
        self.clean_data = clean_data
        self.results = self._evaluate(df_processed=self.df_processed)
        if self.framework_fillna:
            self.results_fillna = self._evaluate(df_processed=self.df_processed, framework_fillna=self.framework_fillna)
        else:
            self.results_fillna = None

    @classmethod
    def from_raw(cls, df_raw: pd.DataFrame, preprocessor_kwargs: dict | None = None, **kwargs):
        from autogluon_benchmark.preprocessing.preprocessor import Preprocessor
        if preprocessor_kwargs is None:
            preprocessor_kwargs = dict()
        df_processed = Preprocessor(df_raw=df_raw, **preprocessor_kwargs).df_processed
        return cls(df_processed=df_processed, **kwargs)

    def _evaluate(
        self,
        df_processed: pd.DataFrame,
        framework_fillna: str | None = None,
    ):
        return evaluate(
            paths=df_processed,
            frameworks_run=self.frameworks,
            framework_nan_fill=framework_fillna,
            task_metadata=self.task_metadata,
            clean_data=self.clean_data,  # FIXME
            folds_to_keep=self.folds,
        )

    def to_plotter(
        self,
        save_dir: str | None = None,
        show: bool = True,
    ) -> Plotter:
        return Plotter(
            results_ranked_df=self.results[1],
            results_ranked_fillna_df=self.results_fillna[1],
            save_dir=save_dir,
            show=show,
        )
