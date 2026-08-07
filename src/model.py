from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "distance_m",
    "gate_number",
    "carried_weight",
    "horse_weight",
    "weight_change",
    "days_since_last",
    "starts_last_180d",
    "avg_finish_last3",
    "top3_rate_last5",
    "distance_top3_rate",
    "surface_top3_rate",
    "jockey_top3_rate",
]
CATEGORICAL_FEATURES = ["sex", "surface", "track_condition"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FINAL_ADDED_FEATURES = [
    "racecourse",
    "field_size",
    "race_class_name",
    "relative_horse_number",
]
FINAL_NUMERIC_FEATURES = [
    *NUMERIC_FEATURES,
    "field_size",
    "relative_horse_number",
]
FINAL_CATEGORICAL_FEATURES = [
    *CATEGORICAL_FEATURES,
    "racecourse",
    "race_class_name",
]
FINAL_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *FINAL_ADDED_FEATURES]
LEAKAGE_COLUMNS = {
    "finish_position",
    "target",
    "race_id",
    "race_date",
    "horse_id",
    "horse_name",
    "jockey_id",
}
REQUIRED_COLUMNS = set(FEATURE_COLUMNS) | LEAKAGE_COLUMNS - {"target"}
FINAL_REQUIRED_COLUMNS = (
    set(FINAL_FEATURE_COLUMNS) | LEAKAGE_COLUMNS - {"target"}
)
MIN_TRAIN_ROWS = 10


class ModelError(ValueError):
    """利用者に日本語で表示できるモデル処理エラー。"""


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    baseline_class: int
    confusion_counts: dict[str, int]
    predictions: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    train_race_ids: tuple[str, ...]
    evaluation_race_ids: tuple[str, ...]


@dataclass
class FinalModelArtifact:
    """固定75%学習領域で学習した正式モデルと分割情報。"""

    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    train_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp


def _classification_metrics(
    actual_target: pd.Series, predicted_target: pd.Series
) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(actual_target, predicted_target),
        "Precision": precision_score(
            actual_target, predicted_target, zero_division=0
        ),
        "Recall": recall_score(actual_target, predicted_target, zero_division=0),
        "F1": f1_score(actual_target, predicted_target, zero_division=0),
    }


def evaluate_majority_baseline(
    train_target: pd.Series, evaluation_target: pd.Series
) -> tuple[int, pd.Series, dict[str, float]]:
    if train_target.empty:
        raise ModelError("多数派Baselineの学習データがありません。")
    if evaluation_target.empty:
        raise ModelError("多数派Baselineの評価データがありません。")

    majority_class = int(train_target.value_counts().idxmax())
    baseline_predictions = pd.Series(
        majority_class,
        index=evaluation_target.index,
        dtype=int,
        name="baseline_predicted_target",
    )
    baseline_metrics = _classification_metrics(
        evaluation_target, baseline_predictions
    )
    return majority_class, baseline_predictions, baseline_metrics


def load_race_csv(
    path: str | PathLike[str],
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError as error:
        raise ModelError(f"データファイルが見つかりません: {path}") from error
    except pd.errors.EmptyDataError as error:
        raise ModelError("CSVが空です。") from error
    except UnicodeDecodeError as error:
        raise ModelError("CSVの文字コードがUTF-8ではありません。") from error
    except (pd.errors.ParserError, OSError) as error:
        raise ModelError(f"CSVの読み込みに失敗しました: {error}") from error

    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ModelError("必要な列が不足しています: " + "、".join(missing_columns))
    if data.empty:
        raise ModelError("CSVにレースデータがありません。")
    return data


def _validate_and_prepare(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ModelError("機械学習に必要な列が不足しています: " + "、".join(missing_columns))
    if data.empty:
        raise ModelError("学習に使用できるデータがありません。")

    prepared = data.copy()
    prepared["race_date"] = pd.to_datetime(prepared["race_date"], errors="coerce")
    if prepared["race_date"].isna().any():
        raise ModelError("race_dateに日付として解釈できない値があります。")

    finish_position = pd.to_numeric(prepared["finish_position"], errors="coerce")
    if finish_position.isna().any():
        raise ModelError("finish_positionに欠損または数値でない値があります。")
    prepared["target"] = (finish_position <= 3).astype(int)
    if prepared["target"].nunique() < 2:
        raise ModelError("finish_positionから作成したtargetが1種類しかありません。")

    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(prepared[column], errors="coerce")
        invalid_values = prepared[column].notna() & converted.isna()
        if invalid_values.any():
            raise ModelError(f"{column}に数値でない値があります。")
        prepared[column] = converted
        if prepared[column].notna().sum() == 0:
            raise ModelError(f"{column}がすべて欠損しているため学習できません。")

    for column in CATEGORICAL_FEATURES:
        if prepared[column].notna().sum() == 0:
            raise ModelError(f"{column}がすべて欠損しているため学習できません。")

    return prepared.sort_values(["race_date", "race_id", "gate_number"]).reset_index(drop=True)


def add_relative_horse_number(data: pd.DataFrame) -> pd.DataFrame:
    """実データ上の馬番（gate_number）を出走頭数で割って追加する。"""
    required = {"gate_number", "field_size"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ModelError(
            "relative_horse_numberの計算に必要な列が不足しています: "
            + "、".join(missing)
        )
    output = data.copy()
    horse_number = pd.to_numeric(output["gate_number"], errors="coerce")
    field_size = pd.to_numeric(output["field_size"], errors="coerce")
    if (output["gate_number"].notna() & horse_number.isna()).any():
        raise ModelError("gate_numberに数値でない値があります。")
    if (output["field_size"].notna() & field_size.isna()).any():
        raise ModelError("field_sizeに数値でない値があります。")
    if field_size.dropna().le(0).any():
        raise ModelError("field_sizeは0より大きい値にしてください。")
    output["relative_horse_number"] = horse_number / field_size
    relative = output["relative_horse_number"].to_numpy(dtype=float)
    if np.isinf(relative).any():
        raise ModelError("relative_horse_numberにinfまたは-infがあります。")
    return output


def prepare_final_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """第3回実験と同じ20特徴量を正式モデル用に検証する。"""
    missing_columns = sorted(FINAL_REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ModelError(
            "正式モデルに必要な列が不足しています: "
            + "、".join(missing_columns)
        )
    prepared = _validate_and_prepare(data)
    for column in ("field_size", "relative_horse_number"):
        converted = pd.to_numeric(prepared[column], errors="coerce")
        invalid = prepared[column].notna() & converted.isna()
        if invalid.any():
            raise ModelError(f"{column}に数値でない値があります。")
        if converted.notna().sum() == 0:
            raise ModelError(f"{column}がすべて欠損しているため学習できません。")
        prepared[column] = converted
    if prepared["field_size"].dropna().le(0).any():
        raise ModelError("field_sizeは0より大きい値にしてください。")
    numeric = prepared[FINAL_NUMERIC_FEATURES].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ModelError("正式モデルの数値特徴量にinfまたは-infがあります。")
    for column in ("racecourse", "race_class_name"):
        if prepared[column].notna().sum() == 0:
            raise ModelError(f"{column}がすべて欠損しているため学習できません。")
    if LEAKAGE_COLUMNS & set(FINAL_FEATURE_COLUMNS):
        raise ModelError("正式モデルの説明変数に使用禁止列が含まれています。")
    return prepared


def _split_by_race_date(
    data: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ModelError("train_ratioは0より大きく1より小さくしてください。")

    unique_dates = data["race_date"].drop_duplicates().sort_values().tolist()
    if len(unique_dates) < 2:
        raise ModelError("時系列分割には異なるrace_dateが2日以上必要です。")

    split_index = int(len(unique_dates) * train_ratio)
    split_index = min(max(split_index, 1), len(unique_dates) - 1)
    train_dates = set(unique_dates[:split_index])
    evaluation_dates = set(unique_dates[split_index:])
    train_data = data[data["race_date"].isin(train_dates)].copy()
    evaluation_data = data[data["race_date"].isin(evaluation_dates)].copy()

    if train_data["race_date"].max() >= evaluation_data["race_date"].min():
        raise ModelError("学習用と評価用データをrace_date順に分割できませんでした。")
    if set(train_data["race_id"]) & set(evaluation_data["race_id"]):
        raise ModelError("同じrace_idが学習用と評価用の両方に含まれています。")
    if len(train_data) < MIN_TRAIN_ROWS:
        raise ModelError(
            f"学習データが少なすぎます。最低{MIN_TRAIN_ROWS}行必要です。"
        )
    if train_data["target"].nunique() < 2:
        raise ModelError("学習用データにtargetの両クラスが必要です。")

    return train_data, evaluation_data


def _build_logistic_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1500, class_weight="balanced"),
            ),
        ]
    )


def _build_pipeline() -> Pipeline:
    if LEAKAGE_COLUMNS & set(FEATURE_COLUMNS):
        raise ModelError("説明変数にデータ漏洩を起こす列が含まれています。")
    return _build_logistic_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def build_final_pipeline() -> Pipeline:
    """第3回Experimentalと同じ正式20特徴量Pipelineを返す。"""
    if LEAKAGE_COLUMNS & set(FINAL_FEATURE_COLUMNS):
        raise ModelError("正式モデルの説明変数に使用禁止列が含まれています。")
    return _build_logistic_pipeline(
        FINAL_NUMERIC_FEATURES,
        FINAL_CATEGORICAL_FEATURES,
    )


def train_final_model_artifact(
    data: pd.DataFrame,
    train_ratio: float = 0.75,
) -> FinalModelArtifact:
    """固定評価の学習期間だけで正式Pipelineを学習する。"""
    prepared = prepare_final_model_data(data)
    train_data, evaluation_data = _split_by_race_date(
        prepared, train_ratio
    )
    pipeline = build_final_pipeline()
    pipeline.fit(
        train_data[FINAL_FEATURE_COLUMNS], train_data["target"]
    )
    return FinalModelArtifact(
        pipeline=pipeline,
        feature_columns=tuple(FINAL_FEATURE_COLUMNS),
        train_rows=len(train_data),
        train_start=train_data["race_date"].min(),
        train_end=train_data["race_date"].max(),
        evaluation_start=evaluation_data["race_date"].min(),
        evaluation_end=evaluation_data["race_date"].max(),
    )


def save_final_model_artifact(
    artifact: FinalModelArtifact,
    path: str | PathLike[str],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(artifact, output)
    except (OSError, ValueError, TypeError) as error:
        raise ModelError(f"正式モデルを保存できませんでした: {error}") from error


def load_final_model_artifact(
    path: str | PathLike[str],
) -> FinalModelArtifact:
    try:
        artifact = joblib.load(path)
    except FileNotFoundError as error:
        raise ModelError(f"正式モデルファイルが見つかりません: {path}") from error
    except (OSError, ValueError, TypeError, EOFError) as error:
        raise ModelError(f"正式モデルを読み込めませんでした: {error}") from error
    if not isinstance(artifact, FinalModelArtifact):
        raise ModelError("正式モデルファイルの形式が正しくありません。")
    if artifact.feature_columns != tuple(FINAL_FEATURE_COLUMNS):
        raise ModelError("保存済みモデルの特徴量構成が20特徴量と一致しません。")
    return artifact


def evaluate_final_model_artifact(
    data: pd.DataFrame,
    artifact: FinalModelArtifact,
    train_ratio: float = 0.75,
) -> EvaluationResult:
    """保存済み正式モデルを、学習時と同じ固定testで評価する。"""
    prepared = prepare_final_model_data(data)
    train_data, evaluation_data = _split_by_race_date(
        prepared, train_ratio
    )
    metadata_matches = (
        artifact.train_rows == len(train_data)
        and artifact.train_start == train_data["race_date"].min()
        and artifact.train_end == train_data["race_date"].max()
        and artifact.evaluation_start == evaluation_data["race_date"].min()
        and artifact.evaluation_end == evaluation_data["race_date"].max()
    )
    if not metadata_matches:
        raise ModelError(
            "保存済み正式モデルと現在のデータ分割が一致しません。"
            "学習スクリプトを再実行してください。"
        )

    model = artifact.pipeline
    predicted_target = model.predict(
        evaluation_data[FINAL_FEATURE_COLUMNS]
    )
    class_index = list(model.named_steps["classifier"].classes_).index(1)
    probabilities = model.predict_proba(
        evaluation_data[FINAL_FEATURE_COLUMNS]
    )[:, class_index]
    metrics = {
        **_classification_metrics(
            evaluation_data["target"], predicted_target
        ),
        "ROC-AUC": roc_auc_score(
            evaluation_data["target"], probabilities
        ),
    }
    baseline_class, baseline_predictions, baseline_metrics = (
        evaluate_majority_baseline(
            train_data["target"], evaluation_data["target"]
        )
    )
    tn, fp, fn, tp = confusion_matrix(
        evaluation_data["target"], predicted_target, labels=[0, 1]
    ).ravel()
    predictions = evaluation_data[
        [
            "race_id",
            "race_date",
            "horse_id",
            "horse_name",
            "jockey_id",
            *FINAL_FEATURE_COLUMNS,
            "finish_position",
            "target",
        ]
    ].copy()
    predictions["predicted_target"] = predicted_target
    predictions["baseline_predicted_target"] = baseline_predictions
    predictions["top3_probability"] = probabilities
    return EvaluationResult(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        baseline_class=baseline_class,
        confusion_counts={
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        },
        predictions=predictions,
        train_start=train_data["race_date"].min(),
        train_end=train_data["race_date"].max(),
        evaluation_start=evaluation_data["race_date"].min(),
        evaluation_end=evaluation_data["race_date"].max(),
        train_race_ids=tuple(train_data["race_id"].astype(str).unique()),
        evaluation_race_ids=tuple(
            evaluation_data["race_id"].astype(str).unique()
        ),
    )


def train_and_evaluate_final(
    data: pd.DataFrame,
    train_ratio: float = 0.75,
) -> EvaluationResult:
    artifact = train_final_model_artifact(data, train_ratio=train_ratio)
    return evaluate_final_model_artifact(
        data, artifact, train_ratio=train_ratio
    )


def train_and_evaluate(
    data: pd.DataFrame, train_ratio: float = 0.75
) -> EvaluationResult:
    prepared = _validate_and_prepare(data)
    train_data, evaluation_data = _split_by_race_date(prepared, train_ratio)

    model = _build_pipeline()
    model.fit(train_data[FEATURE_COLUMNS], train_data["target"])

    predicted_target = model.predict(evaluation_data[FEATURE_COLUMNS])
    class_index = list(model.named_steps["classifier"].classes_).index(1)
    probabilities = model.predict_proba(evaluation_data[FEATURE_COLUMNS])[:, class_index]

    metrics = {
        **_classification_metrics(evaluation_data["target"], predicted_target),
        "ROC-AUC": (
            roc_auc_score(evaluation_data["target"], probabilities)
            if evaluation_data["target"].nunique() == 2
            else float("nan")
        ),
    }
    baseline_class, baseline_predictions, baseline_metrics = (
        evaluate_majority_baseline(
            train_data["target"], evaluation_data["target"]
        )
    )
    tn, fp, fn, tp = confusion_matrix(
        evaluation_data["target"], predicted_target, labels=[0, 1]
    ).ravel()
    confusion_counts = {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }

    predictions = evaluation_data[
        [
            "race_id",
            "race_date",
            "horse_id",
            "horse_name",
            "jockey_id",
            *FEATURE_COLUMNS,
            "finish_position",
            "target",
        ]
    ].copy()
    predictions["predicted_target"] = predicted_target
    predictions["baseline_predicted_target"] = baseline_predictions
    predictions["top3_probability"] = probabilities

    return EvaluationResult(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        baseline_class=baseline_class,
        confusion_counts=confusion_counts,
        predictions=predictions,
        train_start=train_data["race_date"].min(),
        train_end=train_data["race_date"].max(),
        evaluation_start=evaluation_data["race_date"].min(),
        evaluation_end=evaluation_data["race_date"].max(),
        train_race_ids=tuple(train_data["race_id"].astype(str).unique()),
        evaluation_race_ids=tuple(evaluation_data["race_id"].astype(str).unique()),
    )


def select_race_predictions(
    predictions: pd.DataFrame, race_id: str
) -> pd.DataFrame:
    if "race_id" not in predictions.columns:
        raise ModelError("推定結果にrace_id列がありません。")
    selected = predictions[predictions["race_id"].astype(str) == str(race_id)].copy()
    if selected.empty:
        raise ModelError(f"対象レースが存在しません: {race_id}")
    return selected.sort_values("top3_probability", ascending=False)
