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
        frameworks: List[str] | None = None,
        frameworks_compare_vs_all: List[str] | str | None = "auto",
        framework_fillna: str | None = None,
        worst_fillna: bool = False,
        frameworks_rename: Dict[str, str] | None = None,
        task_metadata: pd.DataFrame | None = None,
        clean_data: bool | str = "auto",
        folds: List[int] | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        self.frameworks = frameworks
        self.frameworks_compare_vs_all = frameworks_compare_vs_all
        self.framework_fillna = framework_fillna
        self.worst_fillna = worst_fillna
        self.frameworks_rename = frameworks_rename
        self.task_metadata = task_metadata
        self.folds = folds
        self.verbose = verbose
        self.kwargs = kwargs
        if clean_data == "auto":
            clean_data = self.task_metadata is not None
        self.clean_data = clean_data

    def transform(self, data: pd.DataFrame) -> EvaluatorOutput:
        results = self._evaluate(data=data, **self.kwargs)
        if self.framework_fillna or self.worst_fillna:
            results_fillna = self._evaluate(
                data=data,
                framework_fillna=self.framework_fillna,
                worst_fillna=self.worst_fillna,
                **self.kwargs,
            )
        else:
            results_fillna = None
        return EvaluatorOutput(results=results, results_fillna=results_fillna)

    def _evaluate(
        self,
        data: pd.DataFrame,
        framework_fillna: str | None = None,
        worst_fillna: bool = False,
        **kwargs,
    ):
        return evaluate(
            paths=data,
            frameworks_run=self.frameworks,
            frameworks_compare_vs_all=self.frameworks_compare_vs_all,
            framework_nan_fill=framework_fillna,
            worst_nan_fill=worst_fillna,
            frameworks_rename=self.frameworks_rename,
            task_metadata=self.task_metadata,
            clean_data=self.clean_data,  # FIXME
            folds_to_keep=self.folds,
            verbose=self.verbose,
            **kwargs,
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


class EvaluatorOutput:
    def __init__(self, results, results_fillna=None):
        (
            self.results_ranked_agg,
            self.results_ranked,
            self.results_ranked_all_agg,
            self.results_ranked_all,
            self.results_pairs_merged_dict,
        ) = results
        if results_fillna:
            (
                self.results_fillna_ranked_agg,
                self.results_fillna_ranked,
                _,
                _,
                self.results_fillna_pairs_merged_dict,
            ) = results_fillna
        else:
            self.results_fillna_ranked_agg = None
            self.results_fillna_ranked = None
            self.results_fillna_pairs_merged_dict = None

    @property
    def has_results_fillna(self):
        return self.results_fillna_ranked is not None
