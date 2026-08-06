from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 第1回の追加特徴量は使用せず、モデル生成・評価の共通処理だけを再利用する。
from src.feature_experiment_1 import (
    EXPERIMENT_MODEL_NAMES,
    VERSIONS,
    _baseline_pipeline,
    _evaluate,
    _summarize_cv,
)
from src.model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    ModelError,
    _split_by_race_date,
    _validate_and_prepare,
)
from src.model_comparison import METRIC_COLUMNS
from src.time_series_cv import create_expanding_window_folds


DISTANCE_BAND_FEATURE = "distance_band_top3_rate"
DISTANCE_BAND_BOUNDARIES = (
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
    2200,
    2400,
    2600,
)
DISTANCE_BAND_LABELS = (
    "under_1000",
    "1000_1199",
    "1200_1399",
    "1400_1599",
    "1600_1799",
    "1800_1999",
    "2000_2199",
    "2200_2399",
    "2400_2599",
    "2600_and_over",
)
BASELINE_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
EXPERIMENTAL_FEATURE_COLUMNS = [
    *BASELINE_FEATURE_COLUMNS,
    DISTANCE_BAND_FEATURE,
]
EXPERIMENTAL_NUMERIC_FEATURES = [
    *NUMERIC_FEATURES,
    DISTANCE_BAND_FEATURE,
]


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


def assign_distance_band(distance_m: float | int) -> str:
    try:
        distance = float(distance_m)
    except (TypeError, ValueError) as error:
        raise ModelError(f"distance_mを距離帯へ分類できません: {distance_m}") from error
    if not np.isfinite(distance):
        raise ModelError(f"distance_mを距離帯へ分類できません: {distance_m}")
    band_index = bisect_right(DISTANCE_BAND_BOUNDARIES, distance)
    return DISTANCE_BAND_LABELS[band_index]


def distance_band_distribution(data: pd.DataFrame) -> pd.DataFrame:
    if "distance_m" not in data.columns:
        raise ModelError("距離帯分布の計算にdistance_m列が必要です。")
    bands = data["distance_m"].map(assign_distance_band)
    counts = bands.value_counts().reindex(DISTANCE_BAND_LABELS, fill_value=0)
    return pd.DataFrame(
        {
            "distance_band": DISTANCE_BAND_LABELS,
            "rows": [int(counts[label]) for label in DISTANCE_BAND_LABELS],
            "rate": [
                float(counts[label] / len(data))
                for label in DISTANCE_BAND_LABELS
            ],
        }
    )


def _validate_feature_source(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "race_id",
        "race_date",
        "horse_id",
        "distance_m",
        "gate_number",
        "finish_position",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ModelError(
            "距離帯特徴量の生成に必要な列が不足しています: "
            + "、".join(missing)
        )
    if data.empty:
        raise ModelError("距離帯特徴量を生成するデータがありません。")

    working = data.copy().reset_index(drop=True)
    working["_exp2_original_order"] = np.arange(len(working), dtype=int)
    working["_exp2_race_date"] = pd.to_datetime(
        working["race_date"], errors="coerce"
    )
    race_number = working["race_id"].astype("string").str.extract(
        r"_(\d+)$", expand=False
    )
    working["_exp2_race_number"] = pd.to_numeric(
        race_number, errors="coerce"
    )
    working["_exp2_finish_position"] = pd.to_numeric(
        working["finish_position"], errors="coerce"
    )
    if working[
        [
            "_exp2_race_date",
            "_exp2_race_number",
            "_exp2_finish_position",
            "horse_id",
            "distance_m",
        ]
    ].isna().any().any():
        raise ModelError("距離帯特徴量の時系列計算に必要な値が欠損しています。")
    if not working["_exp2_finish_position"].gt(0).all():
        raise ModelError("実験用入力には有効なfinish_positionが必要です。")
    working["_exp2_distance_band"] = working["distance_m"].map(
        assign_distance_band
    )
    return working


def add_distance_band_top3_rate(data: pd.DataFrame) -> pd.DataFrame:
    """同じ距離帯の対象レースより前の3着内率を追加する。"""
    working = _validate_feature_source(data)
    working = working.sort_values(
        [
            "_exp2_race_date",
            "_exp2_race_number",
            "race_id",
            "gate_number",
            "_exp2_original_order",
        ],
        kind="stable",
    )
    working[DISTANCE_BAND_FEATURE] = np.nan

    history_count: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    top3_count: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # 同日・同race_numberは全行を計算した後で履歴へ追加し、
    # 対象自身と実施順が不明な他場同番号レースの結果を混入させない。
    grouped = working.groupby(
        ["_exp2_race_date", "_exp2_race_number"], sort=False
    )
    for _, time_rows in grouped:
        for index, row in time_rows.iterrows():
            horse_key = str(row["horse_id"])
            band_key = str(row["_exp2_distance_band"])
            count = history_count[horse_key][band_key]
            if count > 0:
                working.at[index, DISTANCE_BAND_FEATURE] = (
                    top3_count[horse_key][band_key] / count
                )

        for _, row in time_rows.iterrows():
            horse_key = str(row["horse_id"])
            band_key = str(row["_exp2_distance_band"])
            history_count[horse_key][band_key] += 1
            if float(row["_exp2_finish_position"]) <= 3:
                top3_count[horse_key][band_key] += 1

    original_columns = list(data.columns)
    base_columns = [
        column for column in original_columns if column != "finish_position"
    ]
    output_columns = [*base_columns, DISTANCE_BAND_FEATURE]
    if "finish_position" in original_columns:
        output_columns.append("finish_position")
    return (
        working.sort_values("_exp2_original_order", kind="stable")
        .reset_index(drop=True)[output_columns]
        .copy()
    )


def validate_experiment_data(
    baseline: pd.DataFrame,
    experimental: pd.DataFrame,
) -> None:
    if DISTANCE_BAND_FEATURE not in experimental.columns:
        raise ModelError(f"{DISTANCE_BAND_FEATURE}がありません。")
    if len(baseline) != len(experimental):
        raise ModelError("BaselineとExperimentalの行数が一致しません。")
    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True),
        experimental[list(baseline.columns)].reset_index(drop=True),
        check_dtype=False,
    )
    values = pd.to_numeric(
        experimental[DISTANCE_BAND_FEATURE], errors="coerce"
    )
    non_missing = values.dropna()
    if not non_missing.between(0.0, 1.0).all():
        raise ModelError(f"{DISTANCE_BAND_FEATURE}が0～1の範囲外です。")


def distance_feature_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    existing_missing_rate = data["distance_top3_rate"].isna().mean()
    for version, feature in (
        ("Existing", "distance_top3_rate"),
        ("Experimental", DISTANCE_BAND_FEATURE),
    ):
        values = pd.to_numeric(data[feature], errors="coerce")
        missing_rate = values.isna().mean()
        rows.append(
            {
                "version": version,
                "feature": feature,
                "min": values.min(),
                "max": values.max(),
                "mean": values.mean(),
                "median": values.median(),
                "missing_count": int(values.isna().sum()),
                "missing_rate": missing_rate,
                "missing_rate_improvement_points": (
                    existing_missing_rate - missing_rate
                ),
            }
        )
    return pd.DataFrame(rows)


def _prepare_experiment_data(data: pd.DataFrame) -> pd.DataFrame:
    if DISTANCE_BAND_FEATURE not in data.columns:
        raise ModelError(f"実験評価に{DISTANCE_BAND_FEATURE}が必要です。")
    prepared = _validate_and_prepare(data)
    prepared[DISTANCE_BAND_FEATURE] = pd.to_numeric(
        prepared[DISTANCE_BAND_FEATURE], errors="coerce"
    )
    if prepared[DISTANCE_BAND_FEATURE].notna().sum() == 0:
        raise ModelError(f"{DISTANCE_BAND_FEATURE}がすべて欠損しています。")
    if set(EXPERIMENTAL_FEATURE_COLUMNS) & LEAKAGE_COLUMNS:
        raise ModelError("Experimental特徴量に使用禁止列が含まれています。")
    return prepared


def _experimental_pipeline(model_name: str) -> Pipeline:
    baseline = _baseline_pipeline(model_name)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, EXPERIMENTAL_NUMERIC_FEATURES),
            (
                "track_condition",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clone(baseline.named_steps["classifier"])),
        ]
    )


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
            raise ModelError("固定25%テストデータが第2回CVへ混入しています。")
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
