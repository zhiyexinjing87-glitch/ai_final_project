from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.model import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    _build_pipeline,
    _classification_metrics,
    _split_by_race_date,
    _validate_and_prepare,
    evaluate_majority_baseline,
)


RANDOM_STATE = 42
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
RESULT_COLUMNS = ["Model", *METRIC_COLUMNS, "Train Rows", "Evaluation Rows"]
CONFUSION_COLUMNS = ["Model", "TN", "FP", "FN", "TP", "Evaluation Rows"]


@dataclass
class ModelComparisonResult:
    results: pd.DataFrame
    confusion_matrices: pd.DataFrame
    feature_columns: tuple[str, ...]
    train_rows: int
    evaluation_rows: int
    train_race_ids: tuple[str, ...]
    evaluation_race_ids: tuple[str, ...]


def _candidate_classifiers() -> dict[str, ClassifierMixin]:
    """比較対象を固定し、乱数を使うモデルには再現可能なseedを設定する。"""
    return {
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_jobs=-1),
    }


def _pipeline_with_classifier(classifier: ClassifierMixin) -> Pipeline:
    """既存ロジスティック回帰と同一の前処理へ比較モデルだけを差し替える。"""
    pipeline = _build_pipeline()
    pipeline.set_params(classifier=classifier)
    return pipeline


def _evaluate_pipeline(
    model: Pipeline,
    train_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
) -> dict[str, float]:
    metrics, _, _ = _evaluate_pipeline_with_predictions(
        model, train_data, evaluation_data
    )
    return metrics


def _evaluate_pipeline_with_predictions(
    model: Pipeline,
    train_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
) -> tuple[dict[str, float], object, object]:
    model.fit(train_data[FEATURE_COLUMNS], train_data["target"])
    predicted_target = model.predict(evaluation_data[FEATURE_COLUMNS])

    classifier = model.named_steps["classifier"]
    class_index = list(classifier.classes_).index(1)
    probabilities = model.predict_proba(evaluation_data[FEATURE_COLUMNS])[
        :, class_index
    ]

    metrics = {
        **_classification_metrics(
            evaluation_data["target"], predicted_target
        ),
        "ROC-AUC": (
            roc_auc_score(evaluation_data["target"], probabilities)
            if evaluation_data["target"].nunique() == 2
            else float("nan")
        ),
    }
    return metrics, predicted_target, probabilities


def _confusion_row(
    model_name: str,
    actual_target: pd.Series,
    predicted_target: object,
) -> dict[str, int | str]:
    tn, fp, fn, tp = confusion_matrix(
        actual_target, predicted_target, labels=[0, 1]
    ).ravel()
    return {
        "Model": model_name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Evaluation Rows": len(actual_target),
    }


def compare_classification_models(
    data: pd.DataFrame,
    train_ratio: float = 0.75,
    candidate_factory: Callable[
        [], dict[str, ClassifierMixin]
    ] = _candidate_classifiers,
    progress_callback: Callable[[str], None] | None = None,
) -> ModelComparisonResult:
    """同じ時系列train/test分割で複数モデルを評価する。"""
    if set(FEATURE_COLUMNS) & LEAKAGE_COLUMNS:
        raise ValueError("説明変数に使用禁止列が含まれています。")

    prepared = _validate_and_prepare(data)
    train_data, evaluation_data = _split_by_race_date(
        prepared, train_ratio
    )
    train_rows = len(train_data)
    evaluation_rows = len(evaluation_data)

    baseline_class, baseline_predictions, baseline_metrics = (
        evaluate_majority_baseline(
        train_data["target"], evaluation_data["target"]
        )
    )
    if progress_callback is not None:
        progress_callback(
            f"Majority Baseline（多数派クラス={baseline_class}）"
        )
    rows = [
        {
            "Model": "Majority Baseline",
            **baseline_metrics,
            "ROC-AUC": float("nan"),
            "Train Rows": train_rows,
            "Evaluation Rows": evaluation_rows,
        }
    ]
    confusion_rows = [
        _confusion_row(
            "Majority Baseline",
            evaluation_data["target"],
            baseline_predictions,
        )
    ]

    # この1行は既存のロジスティック回帰Pipelineをそのまま使用する。
    pipelines = {"Logistic Regression": _build_pipeline()}
    pipelines.update(
        {
            name: _pipeline_with_classifier(classifier)
            for name, classifier in candidate_factory().items()
        }
    )

    for model_name, model in pipelines.items():
        if progress_callback is not None:
            progress_callback(model_name)
        metrics, predicted_target, _ = _evaluate_pipeline_with_predictions(
            model, train_data, evaluation_data
        )
        rows.append(
            {
                "Model": model_name,
                **metrics,
                "Train Rows": train_rows,
                "Evaluation Rows": evaluation_rows,
            }
        )
        confusion_rows.append(
            _confusion_row(
                model_name,
                evaluation_data["target"],
                predicted_target,
            )
        )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return ModelComparisonResult(
        results=results,
        confusion_matrices=pd.DataFrame(
            confusion_rows, columns=CONFUSION_COLUMNS
        ),
        feature_columns=tuple(FEATURE_COLUMNS),
        train_rows=train_rows,
        evaluation_rows=evaluation_rows,
        train_race_ids=tuple(train_data["race_id"].astype(str).unique()),
        evaluation_race_ids=tuple(
            evaluation_data["race_id"].astype(str).unique()
        ),
    )
