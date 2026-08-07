from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.model import ModelError, _split_by_race_date, _validate_and_prepare
from src.time_series_cv import (
    DEFAULT_N_SPLITS,
    TimeSeriesCVResult,
    run_time_series_cross_validation,
)


@dataclass
class FixedTimeSeriesSplit:
    train_data: pd.DataFrame
    test_data: pd.DataFrame


@dataclass
class RealDataTimeSeriesCVResult:
    fixed_split: FixedTimeSeriesSplit
    cross_validation: TimeSeriesCVResult


def create_fixed_time_series_split(
    data: pd.DataFrame,
    train_ratio: float = 0.75,
) -> FixedTimeSeriesSplit:
    prepared = _validate_and_prepare(data)
    train_data, test_data = _split_by_race_date(prepared, train_ratio)
    return FixedTimeSeriesSplit(train_data=train_data, test_data=test_data)


def fixed_split_summary(split: FixedTimeSeriesSplit) -> pd.DataFrame:
    train_data = split.train_data
    test_data = split.test_data
    overlap_count = len(
        set(train_data["race_id"]) & set(test_data["race_id"])
    )
    chronological = bool(
        train_data["race_date"].max() < test_data["race_date"].min()
    )
    return pd.DataFrame(
        [
            {
                "train_start_date": train_data["race_date"]
                .min()
                .date()
                .isoformat(),
                "train_end_date": train_data["race_date"]
                .max()
                .date()
                .isoformat(),
                "test_start_date": test_data["race_date"]
                .min()
                .date()
                .isoformat(),
                "test_end_date": test_data["race_date"]
                .max()
                .date()
                .isoformat(),
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "train_races": train_data["race_id"].nunique(),
                "test_races": test_data["race_id"].nunique(),
                "train_positive_rate": train_data["target"].mean(),
                "test_positive_rate": test_data["target"].mean(),
                "race_id_overlap": overlap_count,
                "train_before_test": chronological,
            }
        ]
    )


def run_real_data_time_series_cv(
    data: pd.DataFrame,
    model_names: tuple[str, ...],
    n_splits: int = DEFAULT_N_SPLITS,
    train_ratio: float = 0.75,
    progress_callback: Callable[[int, str], None] | None = None,
) -> RealDataTimeSeriesCVResult:
    if len(model_names) != 3 or len(set(model_names)) != 3:
        raise ModelError("実データCVの対象モデルは重複なしで3件指定してください。")

    fixed_split = create_fixed_time_series_split(data, train_ratio)
    cross_validation = run_time_series_cross_validation(
        fixed_split.train_data,
        n_splits=n_splits,
        model_names=model_names,
        progress_callback=progress_callback,
    )

    fixed_test_races = set(fixed_split.test_data["race_id"])
    fixed_test_start = fixed_split.test_data["race_date"].min()
    for fold in cross_validation.folds:
        cv_races = set(fold.train_data["race_id"]) | set(
            fold.validation_data["race_id"]
        )
        if cv_races & fixed_test_races:
            raise ModelError("固定25%テストデータがCVへ混入しています。")
        if fold.validation_data["race_date"].max() >= fixed_test_start:
            raise ModelError("CVのvalidation期間が固定テスト期間へ達しています。")

    return RealDataTimeSeriesCVResult(
        fixed_split=fixed_split,
        cross_validation=cross_validation,
    )


def format_real_cv_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Model": "model",
        "Fold": "fold",
        "Train Start": "train_start_date",
        "Train End": "train_end_date",
        "Validation Start": "validation_start_date",
        "Validation End": "validation_end_date",
        "Train Rows": "train_rows",
        "Validation Rows": "validation_rows",
        "Train Races": "train_races",
        "Validation Races": "validation_races",
        "Validation Top3 Rate": "validation_positive_rate",
    }
    rename_map.update(
        {metric: metric.lower().replace("-", "_") for metric in [
            "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
        ]}
    )
    ordered_columns = [
        "model",
        "fold",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "train_rows",
        "validation_rows",
        "train_races",
        "validation_races",
        "validation_positive_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    return fold_results.rename(columns=rename_map)[ordered_columns].copy()


def format_real_cv_summary(summary: pd.DataFrame) -> pd.DataFrame:
    renamed = summary.rename(columns={"Model": "model"}).copy()
    renamed.columns = [
        column.lower().replace("-", "_").replace(" ", "_")
        for column in renamed.columns
    ]
    return renamed
