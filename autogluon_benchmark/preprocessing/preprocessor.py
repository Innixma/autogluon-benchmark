from __future__ import annotations

import pandas as pd

from autogluon.bench.eval.scripts.run_generate_clean_openml import clean_results_df
from autogluon_benchmark.evaluation.evaluator import Evaluator


class Preprocessor:
    def __init__(
        self,
        df_raw: pd.DataFrame,
        framework_suffix: str | None = None,
        framework_suffix_column: str = None,
    ):
        self.framework_suffix = framework_suffix
        self.framework_suffix_column = framework_suffix_column
        self.df_raw = df_raw
        self.df_processed: pd.DataFrame = self.transform(df_raw=self.df_raw)

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        return clean_results_df(
            df_raw=df_raw,
            framework_suffix=self.framework_suffix,
            framework_suffix_column=self.framework_suffix_column,
        )

    def to_evaluator(self, **kwargs) -> Evaluator:
        return Evaluator(df_processed=self.df_processed, **kwargs)
