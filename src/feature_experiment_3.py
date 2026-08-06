from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.feature_experiment_1 import (
    EXPERIMENT_MODEL_NAMES,
    VERSIONS,
    _baseline_pipeline,
    _evaluate,
    _summarize_cv,
)
from src.model import (
    FINAL_ADDED_FEATURES,
    FINAL_CATEGORICAL_FEATURES,
    FINAL_FEATURE_COLUMNS,
    FINAL_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    ModelError,
    _split_by_race_date,
    build_final_pipeline,
    prepare_final_model_data,
)
from src.model_comparison import METRIC_COLUMNS
from src.real_data_converter import (
    assign_column_names,
    clean_base_columns,
    create_direct_features,
    create_race_date,
)
from src.time_series_cv import create_expanding_window_folds


ADDED_FEATURE_COLUMNS = list(FINAL_ADDED_FEATURES)
ADDED_NUMERIC_FEATURES = ["field_size", "relative_horse_number"]
ADDED_CATEGORICAL_FEATURES = ["racecourse", "race_class_name"]
BASELINE_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
EXPERIMENTAL_FEATURE_COLUMNS = list(FINAL_FEATURE_COLUMNS)
EXPERIMENTAL_NUMERIC_FEATURES = list(FINAL_NUMERIC_FEATURES)
EXPERIMENTAL_CATEGORICAL_FEATURES = list(FINAL_CATEGORICAL_FEATURES)

# 元52列に存在しても、レース後に判明するため説明変数へ入れない列。
POST_RACE_COLUMNS = {
    "finish_position",
    "finish_position_dup",
    "time_diff",
    "popularity_rank",
    "running_time_sec",
    "running_time_code",
    "corner_1_position",
    "corner_2_position",
    "corner_3_position",
    "corner_4_position",
    "last_3f",
    "prize_money_10k_yen",
    "pci_like_index",
}
FORBIDDEN_EXPERIMENTAL_COLUMNS = LEAKAGE_COLUMNS | POST_RACE_COLUMNS


@dataclass
class ExperimentSplit:
    train_data: pd.DataFrame
    test_data: pd.DataFrame


@dataclass
class FixedExperimentResult:
    results: pd.DataFrame
    split: ExperimentSplit


@dataclass
class CVExperimentResult:
    fold_results: pd.DataFrame
    summary: pd.DataFrame
    split: ExperimentSplit
    folds: tuple[object, ...]


def extract_race_condition_source(raw_data: pd.DataFrame) -> pd.DataFrame:
    """元52列から、有効行に対応するレース前情報だけを取り出す。"""
    assigned = assign_column_names(raw_data)
    cleaned = clean_base_columns(assigned)
    direct = create_direct_features(create_race_date(cleaned))
    valid_mask = (
        direct["race_status_code"].eq(0)
        & direct["finish_position"].notna()
        & direct["finish_position"].gt(0)
    )
    valid = direct.loc[valid_mask].copy()
    valid["racecourse"] = (
        valid["racecourse"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )
    valid["race_class_name"] = (
        valid["race_class_name"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )
    valid["field_size"] = pd.to_numeric(
        valid["field_size"], errors="coerce"
    )
    valid["horse_number"] = pd.to_numeric(
        valid["horse_number"], errors="coerce"
    )

    nonpositive = valid["field_size"].notna() & valid["field_size"].le(0)
    if nonpositive.any():
        raise ModelError(
            "field_sizeが0以下の有効行があります: "
            f"{int(nonpositive.sum())}行"
        )
    valid["relative_horse_number"] = (
        valid["horse_number"] / valid["field_size"]
    )
    numeric = valid[["field_size", "relative_horse_number"]].to_numpy(
        dtype=float
    )
    if np.isinf(numeric).any():
        raise ModelError("追加特徴量にinfまたは-infが発生しました。")

    source_columns = [
        "race_id",
        "horse_id",
        *ADDED_FEATURE_COLUMNS,
    ]
    source = valid[source_columns].copy()
    if source.duplicated(["race_id", "horse_id"]).any():
        raise ModelError("元52列側にrace_id・horse_idの重複があります。")
    return source.reset_index(drop=True)


def add_race_condition_features(
    baseline: pd.DataFrame,
    feature_source: pd.DataFrame,
) -> pd.DataFrame:
    required = {"race_id", "horse_id", *ADDED_FEATURE_COLUMNS}
    missing = sorted(required - set(feature_source.columns))
    if missing:
        raise ModelError(
            "レース条件特徴量の生成元列が不足しています: "
            + "、".join(missing)
        )
    if baseline.empty:
        raise ModelError("Baselineデータが空です。")

    left = baseline.copy().reset_index(drop=True)
    left["_exp3_order"] = np.arange(len(left), dtype=int)
    left["_exp3_race_id"] = left["race_id"].astype("string")
    left["_exp3_horse_id"] = left["horse_id"].astype("string")

    right = feature_source.copy()
    right["_exp3_race_id"] = right["race_id"].astype("string")
    right["_exp3_horse_id"] = right["horse_id"].astype("string")
    if right.duplicated(["_exp3_race_id", "_exp3_horse_id"]).any():
        raise ModelError("追加特徴量の結合キーが重複しています。")
    right = right[
        ["_exp3_race_id", "_exp3_horse_id", *ADDED_FEATURE_COLUMNS]
    ]

    merged = left.merge(
        right,
        on=["_exp3_race_id", "_exp3_horse_id"],
        how="left",
        sort=False,
        validate="one_to_one",
    ).sort_values("_exp3_order", kind="stable")
    if merged[ADDED_FEATURE_COLUMNS].isna().any().any():
        missing_counts = {
            column: int(merged[column].isna().sum())
            for column in ADDED_FEATURE_COLUMNS
            if merged[column].isna().any()
        }
        raise ModelError(
            f"元52列とBaselineを完全に対応付けできません: {missing_counts}"
        )

    base_columns = [
        column for column in baseline.columns if column != "finish_position"
    ]
    output_columns = [
        *base_columns,
        *ADDED_FEATURE_COLUMNS,
        "finish_position",
    ]
    output = merged[output_columns].reset_index(drop=True).copy()
    validate_experiment_data(baseline, output)
    return output


def validate_experiment_data(
    baseline: pd.DataFrame,
    experimental: pd.DataFrame,
) -> None:
    if len(baseline) != len(experimental):
        raise ModelError("BaselineとExperimentalの行数が一致しません。")
    missing = sorted(set(ADDED_FEATURE_COLUMNS) - set(experimental.columns))
    if missing:
        raise ModelError("追加特徴量が不足しています: " + "、".join(missing))
    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True),
        experimental[list(baseline.columns)].reset_index(drop=True),
        check_dtype=False,
    )

    field_size = pd.to_numeric(experimental["field_size"], errors="coerce")
    relative = pd.to_numeric(
        experimental["relative_horse_number"], errors="coerce"
    )
    horse_number = pd.to_numeric(
        experimental["gate_number"], errors="coerce"
    )
    if field_size.isna().any() or field_size.le(0).any():
        raise ModelError("field_sizeに欠損または0以下の値があります。")
    expected_relative = horse_number / field_size
    if not np.allclose(relative, expected_relative, equal_nan=True):
        raise ModelError("relative_horse_numberの計算が一致しません。")
    if np.isinf(relative.to_numpy(dtype=float)).any():
        raise ModelError("relative_horse_numberにinfまたは-infがあります。")


def additional_feature_statistics(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    for feature in ADDED_CATEGORICAL_FEATURES:
        values = data[feature].astype("string")
        rows.append(
            {
                "feature": feature,
                "kind": "categorical",
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "missing_count": int(values.isna().sum()),
                "unique_count": int(values.nunique(dropna=True)),
                "inf_count": 0,
                "nonpositive_count": 0,
            }
        )
        counts = values.value_counts(dropna=False)
        for category, count in counts.items():
            category_rows.append(
                {
                    "feature": feature,
                    "category": "<missing>" if pd.isna(category) else category,
                    "count": int(count),
                    "rate": float(count / len(data)),
                }
            )

    for feature in ADDED_NUMERIC_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        finite_values = values.replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "feature": feature,
                "kind": "numeric",
                "min": finite_values.min(),
                "max": finite_values.max(),
                "mean": finite_values.mean(),
                "median": finite_values.median(),
                "missing_count": int(values.isna().sum()),
                "unique_count": int(values.nunique(dropna=True)),
                "inf_count": int(np.isinf(values.to_numpy(dtype=float)).sum()),
                "nonpositive_count": int(values.le(0).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(category_rows)


def _prepare_experiment_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(ADDED_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ModelError(
            "第3回実験に必要な追加特徴量が不足しています: "
            + "、".join(missing)
        )
    prepared = prepare_final_model_data(data)
    if set(EXPERIMENTAL_FEATURE_COLUMNS) & FORBIDDEN_EXPERIMENTAL_COLUMNS:
        raise ModelError("Experimental特徴量にレース後情報が含まれています。")
    return prepared


def _experimental_pipeline(model_name: str) -> Pipeline:
    pipeline = build_final_pipeline()
    if model_name == "Logistic Regression":
        return pipeline
    baseline = _baseline_pipeline(model_name)
    pipeline.set_params(
        classifier=clone(baseline.named_steps["classifier"])
    )
    return pipeline


def _pipeline_and_features(
    model_name: str,
    version: str,
) -> tuple[Pipeline, list[str]]:
    if version == "Baseline":
        return _baseline_pipeline(model_name), BASELINE_FEATURE_COLUMNS
    if version == "Experimental":
        return _experimental_pipeline(model_name), EXPERIMENTAL_FEATURE_COLUMNS
    raise ModelError(f"不明な実験versionです: {version}")


def _create_split(data: pd.DataFrame) -> ExperimentSplit:
    prepared = _prepare_experiment_data(data)
    train_data, test_data = _split_by_race_date(prepared, 0.75)
    return ExperimentSplit(train_data=train_data, test_data=test_data)


def run_fixed_feature_experiment(
    data: pd.DataFrame,
    model_names: tuple[str, ...] = EXPERIMENT_MODEL_NAMES,
    progress_callback: Callable[[str, str], None] | None = None,
) -> FixedExperimentResult:
    split = _create_split(data)
    rows = []
    for model_name in model_names:
        for version in VERSIONS:
            if progress_callback is not None:
                progress_callback(model_name, version)
            pipeline, features = _pipeline_and_features(model_name, version)
            metrics = _evaluate(
                pipeline, features, split.train_data, split.test_data
            )
            rows.append(
                {
                    "Model": model_name,
                    "Version": version,
                    **metrics,
                    "Train Rows": len(split.train_data),
                    "Test Rows": len(split.test_data),
                }
            )
    results = pd.DataFrame(rows)
    for metric in METRIC_COLUMNS:
        baseline = (
            results[results["Version"] == "Baseline"]
            .set_index("Model")[metric]
            .to_dict()
        )
        results[f"{metric} Difference"] = results.apply(
            lambda row: row[metric] - baseline[row["Model"]], axis=1
        )
    return FixedExperimentResult(results=results, split=split)


def run_cv_feature_experiment(
    data: pd.DataFrame,
    n_splits: int = 5,
    model_names: tuple[str, ...] = EXPERIMENT_MODEL_NAMES,
    progress_callback: Callable[[int, str, str], None] | None = None,
) -> CVExperimentResult:
    split = _create_split(data)
    folds = create_expanding_window_folds(
        split.train_data, n_splits=n_splits
    )
    fixed_test_races = set(split.test_data["race_id"])
    rows = []
    for fold in folds:
        fold_races = set(fold.train_data["race_id"]) | set(
            fold.validation_data["race_id"]
        )
        if fold_races & fixed_test_races:
            raise ModelError("固定25%テストデータが第3回CVへ混入しています。")
        for model_name in model_names:
            for version in VERSIONS:
                if progress_callback is not None:
                    progress_callback(fold.fold_number, model_name, version)
                pipeline, features = _pipeline_and_features(
                    model_name, version
                )
                metrics = _evaluate(
                    pipeline,
                    features,
                    fold.train_data,
                    fold.validation_data,
                )
                rows.append(
                    {
                        "model": model_name,
                        "version": version,
                        "fold": fold.fold_number,
                        "train_start_date": fold.train_data["race_date"]
                        .min()
                        .date()
                        .isoformat(),
                        "train_end_date": fold.train_data["race_date"]
                        .max()
                        .date()
                        .isoformat(),
                        "validation_start_date": fold.validation_data[
                            "race_date"
                        ]
                        .min()
                        .date()
                        .isoformat(),
                        "validation_end_date": fold.validation_data[
                            "race_date"
                        ]
                        .max()
                        .date()
                        .isoformat(),
                        "train_rows": len(fold.train_data),
                        "validation_rows": len(fold.validation_data),
                        "train_races": fold.train_data["race_id"].nunique(),
                        "validation_races": fold.validation_data[
                            "race_id"
                        ].nunique(),
                        "validation_positive_rate": fold.validation_data[
                            "target"
                        ].mean(),
                        "accuracy": metrics["Accuracy"],
                        "precision": metrics["Precision"],
                        "recall": metrics["Recall"],
                        "f1": metrics["F1"],
                        "roc_auc": metrics["ROC-AUC"],
                    }
                )
    fold_results = pd.DataFrame(rows)
    return CVExperimentResult(
        fold_results=fold_results,
        summary=_summarize_cv(fold_results),
        split=split,
        folds=folds,
    )
