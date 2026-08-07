from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    ModelError,
    _classification_metrics,
    _split_by_race_date,
    _validate_and_prepare,
)
from src.model_comparison import (
    METRIC_COLUMNS,
    _candidate_classifiers,
    _pipeline_with_classifier,
)
from src.time_series_cv import create_expanding_window_folds


ADDED_FEATURE_COLUMNS = [
    "horse_history_count",
    "distance_history_count",
    "surface_history_count",
    "jockey_history_count",
    "has_previous_race",
    "has_distance_history",
    "has_surface_history",
]
COUNT_FEATURE_COLUMNS = ADDED_FEATURE_COLUMNS[:4]
FLAG_FEATURE_COLUMNS = ADDED_FEATURE_COLUMNS[4:]
BASELINE_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
EXPERIMENTAL_FEATURE_COLUMNS = [
    *BASELINE_FEATURE_COLUMNS,
    *ADDED_FEATURE_COLUMNS,
]
EXPERIMENTAL_NUMERIC_FEATURES = [
    *NUMERIC_FEATURES,
    *ADDED_FEATURE_COLUMNS,
]
EXPERIMENT_MODEL_NAMES = (
    "Logistic Regression",
    "Gradient Boosting",
    "AdaBoost",
)
VERSIONS = ("Baseline", "Experimental")


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


def _validate_count_source(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "race_id",
        "race_date",
        "horse_id",
        "jockey_id",
        "distance_m",
        "surface",
        "gate_number",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ModelError(
            "履歴量特徴量の生成に必要な列が不足しています: "
            + "、".join(missing)
        )
    if data.empty:
        raise ModelError("履歴量特徴量を生成するデータがありません。")

    working = data.copy().reset_index(drop=True)
    working["_exp_original_order"] = np.arange(len(working), dtype=int)
    working["_exp_race_date"] = pd.to_datetime(
        working["race_date"], errors="coerce"
    )
    if working["_exp_race_date"].isna().any():
        raise ModelError("race_dateに日付として解釈できない値があります。")

    race_number = working["race_id"].astype("string").str.extract(
        r"_(\d+)$", expand=False
    )
    working["_exp_race_number"] = pd.to_numeric(
        race_number, errors="coerce"
    )
    if working["_exp_race_number"].isna().any():
        raise ModelError("race_idからrace_numberを取得できない行があります。")

    working["_exp_distance_m"] = pd.to_numeric(
        working["distance_m"], errors="coerce"
    )
    required_values = [
        "horse_id",
        "jockey_id",
        "surface",
        "_exp_distance_m",
    ]
    if working[required_values].isna().any().any():
        raise ModelError("履歴量特徴量の識別子または条件列に欠損があります。")
    return working


def add_history_count_features(data: pd.DataFrame) -> pd.DataFrame:
    """対象レースより前の有効履歴だけから7特徴量を追加する。"""
    working = _validate_count_source(data)
    working = working.sort_values(
        [
            "_exp_race_date",
            "_exp_race_number",
            "race_id",
            "gate_number",
            "_exp_original_order",
        ],
        kind="stable",
    )

    for column in COUNT_FEATURE_COLUMNS:
        working[column] = np.int64(0)

    horse_counts: dict[str, int] = defaultdict(int)
    distance_counts: dict[str, dict[float, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    surface_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    jockey_counts: dict[str, int] = defaultdict(int)

    # 既存変換器と同じく、同日・同race_numberは全行を計算してから
    # 履歴へ追加する。実施順が断定できない他場同番号レースを混入させない。
    grouped = working.groupby(
        ["_exp_race_date", "_exp_race_number"], sort=False
    )
    for _, time_rows in grouped:
        for index, row in time_rows.iterrows():
            horse_key = str(row["horse_id"])
            jockey_key = str(row["jockey_id"])
            distance_key = float(row["_exp_distance_m"])
            surface_key = str(row["surface"])
            working.at[index, "horse_history_count"] = horse_counts[
                horse_key
            ]
            working.at[index, "distance_history_count"] = distance_counts[
                horse_key
            ][distance_key]
            working.at[index, "surface_history_count"] = surface_counts[
                horse_key
            ][surface_key]
            working.at[index, "jockey_history_count"] = jockey_counts[
                jockey_key
            ]

        for _, row in time_rows.iterrows():
            horse_key = str(row["horse_id"])
            jockey_key = str(row["jockey_id"])
            distance_key = float(row["_exp_distance_m"])
            surface_key = str(row["surface"])
            horse_counts[horse_key] += 1
            distance_counts[horse_key][distance_key] += 1
            surface_counts[horse_key][surface_key] += 1
            jockey_counts[jockey_key] += 1

    working["has_previous_race"] = (
        working["horse_history_count"] > 0
    ).astype(int)
    working["has_distance_history"] = (
        working["distance_history_count"] > 0
    ).astype(int)
    working["has_surface_history"] = (
        working["surface_history_count"] > 0
    ).astype(int)

    original_columns = list(data.columns)
    finish_at_end = "finish_position" in original_columns
    base_columns = [
        column for column in original_columns if column != "finish_position"
    ]
    output_columns = [*base_columns, *ADDED_FEATURE_COLUMNS]
    if finish_at_end:
        output_columns.append("finish_position")
    return (
        working.sort_values("_exp_original_order", kind="stable")
        .reset_index(drop=True)[output_columns]
        .copy()
    )


def validate_experiment_data(
    baseline: pd.DataFrame,
    experimental: pd.DataFrame,
) -> dict[str, int]:
    missing_features = sorted(
        set(ADDED_FEATURE_COLUMNS) - set(experimental.columns)
    )
    if missing_features:
        raise ModelError(
            "実験用データに追加特徴量が不足しています: "
            + "、".join(missing_features)
        )
    if len(baseline) != len(experimental):
        raise ModelError("BaselineとExperimentalの行数が一致しません。")
    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True),
        experimental[list(baseline.columns)].reset_index(drop=True),
        check_dtype=False,
    )

    for column in COUNT_FEATURE_COLUMNS:
        values = pd.to_numeric(experimental[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ModelError(f"{column}に欠損または負の値があります。")
        if not np.equal(values, np.floor(values)).all():
            raise ModelError(f"{column}に整数でない値があります。")
    for column in FLAG_FEATURE_COLUMNS:
        if not experimental[column].isin([0, 1]).all():
            raise ModelError(f"{column}は0または1である必要があります。")

    checks = {
        "previous_history_mismatch": int(
            (
                experimental["has_previous_race"].eq(0)
                != experimental["avg_finish_last3"].isna()
            ).sum()
        ),
        "distance_history_mismatch": int(
            (
                experimental["has_distance_history"].eq(0)
                != experimental["distance_top3_rate"].isna()
            ).sum()
        ),
        "surface_history_mismatch": int(
            (
                experimental["has_surface_history"].eq(0)
                != experimental["surface_top3_rate"].isna()
            ).sum()
        ),
        "jockey_history_mismatch": int(
            (
                experimental["jockey_history_count"].eq(0)
                != experimental["jockey_top3_rate"].isna()
            ).sum()
        ),
    }
    if any(checks.values()):
        raise ModelError(f"履歴有無と既存履歴特徴量が不整合です: {checks}")

    expected_flags = {
        "has_previous_race": "horse_history_count",
        "has_distance_history": "distance_history_count",
        "has_surface_history": "surface_history_count",
    }
    for flag, count in expected_flags.items():
        if not experimental[flag].eq(experimental[count].gt(0).astype(int)).all():
            raise ModelError(f"{flag}と{count}が整合しません。")
    return checks


def additional_feature_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ADDED_FEATURE_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "min": values.min(),
                "max": values.max(),
                "mean": values.mean(),
                "median": values.median(),
                "missing_count": int(values.isna().sum()),
                "missing_rate": values.isna().mean(),
                "zero_count": int(values.eq(0).sum()),
                "zero_rate": values.eq(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def _prepare_experiment_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(ADDED_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ModelError(
            "実験評価に必要な追加特徴量が不足しています: "
            + "、".join(missing)
        )
    prepared = _validate_and_prepare(data)
    for column in ADDED_FEATURE_COLUMNS:
        converted = pd.to_numeric(prepared[column], errors="coerce")
        if converted.isna().any():
            raise ModelError(f"{column}に欠損または数値でない値があります。")
        prepared[column] = converted
    if set(EXPERIMENTAL_FEATURE_COLUMNS) & LEAKAGE_COLUMNS:
        raise ModelError("Experimental特徴量に使用禁止列が含まれています。")
    return prepared


def _baseline_pipeline(model_name: str) -> Pipeline:
    if model_name == "Logistic Regression":
        from src.model import _build_pipeline

        return _build_pipeline()
    candidates = _candidate_classifiers()
    if model_name not in candidates:
        raise ModelError(f"実験対象外のモデルです: {model_name}")
    return _pipeline_with_classifier(candidates[model_name])


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


def _evaluate(
    pipeline: Pipeline,
    feature_columns: list[str],
    train_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
) -> dict[str, float]:
    pipeline.fit(train_data[feature_columns], train_data["target"])
    predictions = pipeline.predict(evaluation_data[feature_columns])
    classifier = pipeline.named_steps["classifier"]
    class_index = list(classifier.classes_).index(1)
    probabilities = pipeline.predict_proba(
        evaluation_data[feature_columns]
    )[:, class_index]
    return {
        **_classification_metrics(evaluation_data["target"], predictions),
        "ROC-AUC": roc_auc_score(
            evaluation_data["target"], probabilities
        ),
    }


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
        baseline_by_model = (
            results[results["Version"] == "Baseline"]
            .set_index("Model")[metric]
            .to_dict()
        )
        results[f"{metric} Difference"] = results.apply(
            lambda row: row[metric] - baseline_by_model[row["Model"]],
            axis=1,
        )
    return FixedExperimentResult(results=results, split=split)


def _summarize_cv(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in fold_results["model"].drop_duplicates():
        model_rows = fold_results[fold_results["model"] == model_name]
        baseline = model_rows[model_rows["version"] == "Baseline"]
        for version in VERSIONS:
            selected = model_rows[model_rows["version"] == version]
            row: dict[str, str | float] = {
                "model": model_name,
                "version": version,
            }
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                values = selected[metric].astype(float)
                baseline_values = baseline[metric].astype(float)
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=0)
                row[f"{metric}_mean_difference"] = (
                    values.mean() - baseline_values.mean()
                )
                row[f"{metric}_std_difference"] = (
                    values.std(ddof=0) - baseline_values.std(ddof=0)
                )
            rows.append(row)
    return pd.DataFrame(rows)


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
            raise ModelError("固定25%テストデータが実験CVへ混入しています。")
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
