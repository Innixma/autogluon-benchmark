from __future__ import annotations

from typing import Dict

import pandas as pd

from autogluon.bench.eval.scripts.run_generate_clean_openml import clean_results_df


class AMLBPreprocessor:
    def __init__(
        self,
        framework_suffix: str | None = None,
        framework_suffix_column: str | None = "constraint",
        framework_rename_dict: Dict[str, str] | None = None,
    ):
        """
        Preprocessor class that is used to transform data in AMLB format to AG-Bench format.

        Parameters
        ----------
        framework_suffix: str, default = None
            If specified, value will be appended to the `framework` column in the format "{framework}_{framework_suffix}"
        framework_suffix_column: str, default = "constraint"
            The column name to use for the framework_suffix. Note that this can be specified alongside `framework_suffix`.
            When both are specified, the result will be "{framework}_{framework_suffix_column_value}_{framework_suffix}"
        framework_rename_dict: dict, default = None
            If specified, will rename the frameworks prior to updating their names with other logic based on this mapping.
        """
        self.framework_suffix = framework_suffix
        self.framework_suffix_column = framework_suffix_column
        self.framework_rename_dict = framework_rename_dict

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data in AMLB format to AG-Bench format.

        Parameters
        ----------
        df: pd.DataFrame
            The raw data in AMLB format.

            Required Columns:
            task
                Dataset name. Provided by AMLB.
            fold
                Fold number. Provided by AMLB.
            framework
                Framework name. Provided by AMLB.
            result
                Metric score. Provided by AMLB.
                Used in combination with `metric` to compute `metric_error`.
            metric
                Metric as a string. Provided by AMLB.
                Used in combination with `result` to compute `metric_error`.
            type
                Problem type of the task. Provided by AMLB.
            training_duration
                Train time in seconds. Provided by AMLB.
            predict_duration
                Infer time in seconds. Provided by AMLB.
            id
                The task id as a string. Provided by AMLB.

        Returns
        -------
        df_transform: pd.DataFrame
            The processed data in AG-Bench format.

            Required Columns:
            dataset
                Dataset name.
            fold
                Fold number.
            framework
                Framework name.
            metric_error
                Metric error as a float.
            metric
                Metric name as a string.
            problem_type
                Problem type of the task
            time_train_s
                Train time in seconds.
            time_infer_s
                Infer time in seconds.
            tid
                Task ID as an integer.
        """
        return clean_results_df(
            df_raw=df,
            framework_suffix=self.framework_suffix,
            framework_suffix_column=self.framework_suffix_column,
            framework_rename_dict=self.framework_rename_dict,
        )
