from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.model import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    ModelError,
    _build_pipeline,
    _validate_and_prepare,
)
from src.model_comparison import (
    METRIC_COLUMNS,
    _candidate_classifiers,
    _evaluate_pipeline,
    _pipeline_with_classifier,
)


DEFAULT_N_SPLITS = 5
CV_MODEL_NAMES = (
    "Logistic Regression",
    "Gradient Boosting",
    "AdaBoost",
)
FOLD_RESULT_COLUMNS = [
    "Model",
    "Fold",
    *METRIC_COLUMNS,
    "Train Start",
    "Train End",
    "Validation Start",
    "Validation End",
    "Train Rows",
    "Validation Rows",
    "Train Races",
    "Validation Races",
    "Validation Top3 Rate",
]
SUMMARY_STATISTICS = ("Mean", "Std", "Min", "Max")
SUMMARY_COLUMNS = [
    "Model",
    *[
        f"{metric} {statistic}"
        for metric in METRIC_COLUMNS
        for statistic in SUMMARY_STATISTICS
    ],
]


@dataclass
class ExpandingWindowFold:
    fold_number: int
    train_data: pd.DataFrame
    validation_data: pd.DataFrame


@dataclass
class TimeSeriesCVResult:
    fold_results: pd.DataFrame
    summary: pd.DataFrame
    folds: tuple[ExpandingWindowFold, ...]
    feature_columns: tuple[str, ...]


def create_expanding_window_folds(
    data: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[ExpandingWindowFold, ...]:
    """日付ブロック単位でtrainを広げる時系列foldを作る。"""
    if n_splits < 2:
        raise ModelError("時系列交差検証のfold数は2以上にしてください。")

    prepared = _validate_and_prepare(data)
    unique_dates = np.array(
        sorted(prepared["race_date"].unique()),
        dtype="datetime64[ns]",
    )
    if len(unique_dates) < n_splits + 1:
        raise ModelError(
            "時系列交差検証に必要なrace_date数が不足しています。"
        )

    date_blocks = np.array_split(unique_dates, n_splits + 1)
    if any(len(block) == 0 for block in date_blocks):
        raise ModelError("空の期間を含む時系列foldは作成できません。")

    folds: list[ExpandingWindowFold] = []
    for fold_number in range(1, n_splits + 1):
        train_dates = np.concatenate(date_blocks[:fold_number])
        validation_dates = date_blocks[fold_number]
        train_data = prepared[
            prepared["race_date"].isin(train_dates)
        ].copy()
        validation_data = prepared[
            prepared["race_date"].isin(validation_dates)
        ].copy()

        if train_data["race_date"].max() >= validation_data["race_date"].min():
            raise ModelError(
                f"Fold {fold_number}を日付順に分割できませんでした。"
            )
        if set(train_data["race_id"]) & set(validation_data["race_id"]):
            raise ModelError(
                f"Fold {fold_number}のtrainとvalidationに同じrace_idがあります。"
            )
        if train_data["target"].nunique() < 2:
            raise ModelError(
                f"Fold {fold_number}のtrainにtargetの両クラスが必要です。"
            )
        if validation_data["target"].nunique() < 2:
            raise ModelError(
                f"Fold {fold_number}のvalidationにtargetの両クラスが必要です。"
            )

        folds.append(
            ExpandingWindowFold(
                fold_number=fold_number,
                train_data=train_data,
                validation_data=validation_data,
            )
        )

    validation_date_sets = [
        set(fold.validation_data["race_date"]) for fold in folds
    ]
    for index, current_dates in enumerate(validation_date_sets):
        for later_dates in validation_date_sets[index + 1 :]:
            if current_dates & later_dates:
                raise ModelError("validation期間同士が重複しています。")

    return tuple(folds)


def _time_series_model_pipelines(
    model_names: tuple[str, ...] = CV_MODEL_NAMES,
) -> dict[str, object]:
    candidates = _candidate_classifiers()
    available = {
        "Logistic Regression": _build_pipeline(),
        **{
            name: _pipeline_with_classifier(classifier)
            for name, classifier in candidates.items()
        },
    }
    unknown_models = sorted(set(model_names) - set(available))
    if unknown_models:
        raise ModelError(
            "時系列交差検証で利用できないモデルです: "
            + "、".join(unknown_models)
        )
    if not model_names or len(set(model_names)) != len(model_names):
        raise ModelError("CV対象モデルは重複なしで1件以上指定してください。")
    return {name: available[name] for name in model_names}


def summarize_fold_results(
    fold_results: pd.DataFrame,
    model_names: tuple[str, ...] = CV_MODEL_NAMES,
) -> pd.DataFrame:
    missing_columns = sorted(
        {"Model", *METRIC_COLUMNS} - set(fold_results.columns)
    )
    if missing_columns:
        raise ModelError(
            "fold結果の集計に必要な列が不足しています: "
            + "、".join(missing_columns)
        )

    summary_rows = []
    for model_name in model_names:
        model_results = fold_results[
            fold_results["Model"] == model_name
        ]
        if model_results.empty:
            raise ModelError(f"{model_name}のfold結果がありません。")

        row: dict[str, str | float] = {"Model": model_name}
        for metric_name in METRIC_COLUMNS:
            values = model_results[metric_name].astype(float)
            row[f"{metric_name} Mean"] = values.mean()
            row[f"{metric_name} Std"] = values.std(ddof=0)
            row[f"{metric_name} Min"] = values.min()
            row[f"{metric_name} Max"] = values.max()
        summary_rows.append(row)

    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def run_time_series_cross_validation(
    data: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    model_names: tuple[str, ...] = CV_MODEL_NAMES,
    progress_callback: Callable[[int, str], None] | None = None,
) -> TimeSeriesCVResult:
    """指定3モデルをexpanding-window方式で評価する。"""
    if set(FEATURE_COLUMNS) & LEAKAGE_COLUMNS:
        raise ModelError("説明変数に使用禁止列が含まれています。")

    folds = create_expanding_window_folds(data, n_splits=n_splits)
    rows = []
    for fold in folds:
        for model_name, model in _time_series_model_pipelines(
            model_names
        ).items():
            if progress_callback is not None:
                progress_callback(fold.fold_number, model_name)
            metrics = _evaluate_pipeline(
                model,
                fold.train_data,
                fold.validation_data,
            )
            rows.append(
                {
                    "Model": model_name,
                    "Fold": fold.fold_number,
                    **metrics,
                    "Train Start": fold.train_data["race_date"]
                    .min()
                    .date()
                    .isoformat(),
                    "Train End": fold.train_data["race_date"]
                    .max()
                    .date()
                    .isoformat(),
                    "Validation Start": fold.validation_data["race_date"]
                    .min()
                    .date()
                    .isoformat(),
                    "Validation End": fold.validation_data["race_date"]
                    .max()
                    .date()
                    .isoformat(),
                    "Train Rows": len(fold.train_data),
                    "Validation Rows": len(fold.validation_data),
                    "Train Races": fold.train_data["race_id"].nunique(),
                    "Validation Races": fold.validation_data[
                        "race_id"
                    ].nunique(),
                    "Validation Top3 Rate": fold.validation_data[
                        "target"
                    ].mean(),
                }
            )

    fold_results = pd.DataFrame(rows, columns=FOLD_RESULT_COLUMNS)
    summary = summarize_fold_results(
        fold_results, model_names=model_names
    )
    return TimeSeriesCVResult(
        fold_results=fold_results,
        summary=summary,
        folds=folds,
        feature_columns=tuple(FEATURE_COLUMNS),
    )
