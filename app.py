from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai_explainer import (
    ExplanationContext,
    build_ai_review,
    explain_model_results,
    explain_model_results_with_gemini,
)
from src.model import (
    FINAL_FEATURE_COLUMNS,
    FINAL_REQUIRED_COLUMNS,
    ModelError,
    evaluate_final_model_artifact,
    load_final_model_artifact,
    load_race_csv,
    select_race_predictions,
)
from src.report_generator import generate_markdown_report


PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "model_ready_real_data_exp3.csv"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final_logistic_regression.pkl"
MODEL_COMPARISON_PATH = (
    PROJECT_ROOT / "results" / "feature_experiment_3_fixed.csv"
)
TIME_SERIES_CV_FOLDS_PATH = (
    PROJECT_ROOT / "results" / "feature_experiment_3_cv_folds.csv"
)
TIME_SERIES_CV_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "feature_experiment_3_cv_summary.csv"
)
FEATURE_COLUMNS = FINAL_FEATURE_COLUMNS
REQUIRED_COLUMNS = FINAL_REQUIRED_COLUMNS
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


@st.cache_data(show_spinner=False)
def load_app_data(path: str) -> pd.DataFrame:
    return load_race_csv(path, REQUIRED_COLUMNS)


@st.cache_resource(show_spinner=False)
def load_app_model(path: str):
    return load_final_model_artifact(path)


@st.cache_data(show_spinner=False)
def load_saved_result(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def race_number_from_id(race_id: object) -> str:
    """race_id末尾のレース番号を画面表示用に取り出す。"""
    value = str(race_id).rsplit("_", maxsplit=1)[-1]
    return str(int(value)) if value.isdigit() else value


def display_value(value: object, suffix: str = "") -> str:
    if pd.isna(value):
        return "不明"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{suffix}"


def read_formal_metrics(fixed_results: pd.DataFrame) -> dict[str, float]:
    required = {"Model", "Version", *METRIC_NAMES}
    missing = sorted(required - set(fixed_results.columns))
    if missing:
        raise ValueError("必要な列がありません: " + "、".join(missing))
    row = fixed_results[
        fixed_results["Model"].eq("Logistic Regression")
        & fixed_results["Version"].eq("Experimental")
    ]
    if len(row) != 1:
        raise ValueError(
            "20特徴量版Logistic Regressionの評価結果を1件に特定できません。"
        )
    return {name: float(row.iloc[0][name]) for name in METRIC_NAMES}


st.set_page_config(
    page_title="競馬レースデータ分析AI",
    page_icon="🏇",
    layout="wide",
)
st.title("競馬レースデータ分析AI")
st.caption(
    "過去レースデータと機械学習を利用し、各出走馬の3着以内となる傾向を"
    "分析する授業・研究用アプリ"
)
st.caption(
    "使用モデル：Logistic Regression　｜　特徴量：20項目　｜　"
    "データ：約2年半"
)
st.warning(
    "本アプリは授業・研究目的の過去データ分析です。"
    "実際の競走結果を保証するものではありません。"
)

try:
    races = load_app_data(str(DATA_PATH))
except ModelError as error:
    st.error(str(error))
    st.stop()

try:
    final_model = load_app_model(str(FINAL_MODEL_PATH))
    result = evaluate_final_model_artifact(races, final_model)
except ModelError as error:
    st.error(f"モデル処理を実行できませんでした: {error}")
    st.stop()
except Exception as error:
    st.error(f"予期しないエラーによりモデル処理に失敗しました: {error}")
    st.stop()

fixed_results: pd.DataFrame | None = None
formal_metrics: dict[str, float] | None = None
try:
    fixed_results = load_saved_result(str(MODEL_COMPARISON_PATH))
    formal_metrics = read_formal_metrics(fixed_results)
except FileNotFoundError:
    st.warning(
        "固定テストの評価結果ファイルが見つかりません。"
        f"{MODEL_COMPARISON_PATH.name}を確認してください。"
    )
except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
    st.warning(f"固定テストの評価結果を読み込めませんでした: {error}")

st.sidebar.header("アプリ情報")
st.sidebar.write("**対象**：終了済みの過去レース")
st.sidebar.write("**正式モデル**：Logistic Regression")
st.sidebar.write("**使用特徴量**：20項目")
st.sidebar.write(f"**収録データ**：{len(races):,}行")
st.sidebar.caption(
    f"評価期間：{result.evaluation_start.date()} ～ "
    f"{result.evaluation_end.date()}"
)
st.sidebar.info("授業・研究用の分析画面です。")


# 固定テストの推定結果から、日付・競馬場・レースの順に対象を絞り込む。
selection_data = result.predictions.copy()
selection_data["_race_date_display"] = pd.to_datetime(
    selection_data["race_date"], errors="coerce"
).dt.strftime("%Y-%m-%d")
selection_data["_race_number_display"] = selection_data["race_id"].map(
    race_number_from_id
)
selection_data["_race_number_sort"] = pd.to_numeric(
    selection_data["_race_number_display"], errors="coerce"
)

st.header("1. 分析するレースを選択")
selection_columns = st.columns(3)
date_options = sorted(selection_data["_race_date_display"].dropna().unique())
selected_date = selection_columns[0].selectbox("1. 開催日", date_options)

date_rows = selection_data[
    selection_data["_race_date_display"].eq(selected_date)
]
course_options = sorted(date_rows["racecourse"].dropna().astype(str).unique())
selected_course = selection_columns[1].selectbox("2. 競馬場", course_options)

race_options = (
    date_rows[date_rows["racecourse"].astype(str).eq(selected_course)]
    .sort_values(["_race_number_sort", "race_id"])
    .drop_duplicates("race_id")
)
race_ids = race_options["race_id"].astype(str).tolist()
race_labels = {
    str(row["race_id"]): (
        f"{row['_race_date_display']}｜{row['racecourse']}｜"
        f"{row['_race_number_display']}R｜{row['race_class_name']}"
    )
    for _, row in race_options.iterrows()
}
selected_race_id = selection_columns[2].selectbox(
    "3. レース",
    race_ids,
    format_func=lambda race_id: race_labels.get(str(race_id), str(race_id)),
)

try:
    race_predictions = select_race_predictions(
        result.predictions, str(selected_race_id)
    )
except ModelError as error:
    st.error(str(error))
    st.stop()


st.header("2. 選択レースの基本情報")
race = race_predictions.iloc[0]
race_date = pd.to_datetime(race["race_date"], errors="coerce")
race_date_text = race_date.date().isoformat() if not pd.isna(race_date) else "不明"
first_info_row = st.columns(4)
first_info_row[0].metric("開催日", race_date_text)
first_info_row[1].metric("競馬場", display_value(race["racecourse"]))
first_info_row[2].metric(
    "レース番号", f"{race_number_from_id(selected_race_id)}R"
)
first_info_row[3].metric(
    "レースクラス", display_value(race["race_class_name"])
)
second_info_row = st.columns(4)
second_info_row[0].metric("距離", display_value(race["distance_m"], "m"))
second_info_row[1].metric("芝・ダート", display_value(race["surface"]))
second_info_row[2].metric("馬場状態", display_value(race["track_condition"]))
second_info_row[3].metric("出走頭数", display_value(race["field_size"], "頭"))


st.header("3. モデル分析結果")
st.caption(
    "3着以内となる傾向の推定値です。推定確率が高い順に表示しています。"
)
display_predictions = race_predictions[
    [
        "gate_number",
        "horse_name",
        "top3_probability",
        "finish_position",
        "jockey_id",
    ]
].rename(
    columns={
        "gate_number": "馬番",
        "horse_name": "馬名",
        "top3_probability": "推定確率",
        "finish_position": "実際の着順",
        "jockey_id": "騎手ID",
    }
)
st.dataframe(
    display_predictions,
    hide_index=True,
    width="stretch",
    column_config={
        "推定確率": st.column_config.ProgressColumn(
            "推定確率",
            min_value=0.0,
            max_value=1.0,
            format="percent",
        )
    },
)

st.subheader("推定確率の可視化")
probability_chart = race_predictions[
    ["gate_number", "horse_name", "top3_probability"]
].copy()
probability_chart["馬名"] = (
    probability_chart["gate_number"].astype(int).astype(str)
    + "番 "
    + probability_chart["horse_name"].astype(str)
)
probability_chart = probability_chart.set_index("馬名")[["top3_probability"]]
probability_chart = probability_chart.rename(
    columns={"top3_probability": "推定確率"}
)
st.bar_chart(probability_chart, height=360, width="stretch")


with st.expander("モデル情報"):
    st.markdown(
        """
- 正式モデル：Logistic Regression
- 使用特徴量：20項目
- クラス重み：`class_weight="balanced"`
- 分類閾値：`0.5`
- 数値特徴量：中央値補完＋標準化
- カテゴリ特徴量：最頻値補完＋One-Hot Encoding
- 未知カテゴリ：`handle_unknown="ignore"`
"""
    )
    st.caption(
        "finish_position、target、race_id、race_date、horse_id、horse_name、"
        "jockey_idは説明変数に使用していません。"
    )


st.header("4. 正式モデルの固定テスト評価")
st.caption(
    "古い75%を学習、新しい25%を固定テストとして評価した保存済み結果です。"
)
if formal_metrics is not None:
    metric_columns = st.columns(len(METRIC_NAMES))
    for column, metric_name in zip(metric_columns, METRIC_NAMES):
        column.metric(metric_name, f"{formal_metrics[metric_name]:.3f}")
    st.caption(
        f"学習期間：{result.train_start.date()} ～ {result.train_end.date()}　｜　"
        f"評価期間：{result.evaluation_start.date()} ～ "
        f"{result.evaluation_end.date()}　｜　"
        f"出典：{MODEL_COMPARISON_PATH.name}（Experimental）"
    )


with st.expander("固定テスト評価の詳細（Baseline・混同行列）"):
    baseline_label = (
        "3着以内（target=1）"
        if result.baseline_class == 1
        else "4着以下（target=0）"
    )
    st.write("#### Majority Baselineとの比較")
    st.caption(
        f"学習データで最も多かった「{baseline_label}」を"
        "評価データの全件に予測する単純な基準です。"
    )
    comparison_table = pd.DataFrame(
        [
            {
                "Model": "Majority Baseline",
                **result.baseline_metrics,
                "ROC-AUC": float("nan"),
            },
            {
                "Model": "Logistic Regression",
                **(formal_metrics or result.metrics),
            },
        ]
    )
    st.dataframe(
        comparison_table,
        hide_index=True,
        width="stretch",
        column_config={
            metric_name: st.column_config.NumberColumn(
                metric_name, format="%.3f"
            )
            for metric_name in METRIC_NAMES
        },
    )
    st.caption(
        "Majority Baselineは全件に同じクラスを返し、確率の順位付けを"
        "しないため、ROC-AUCは算出していません。"
    )

    st.write("#### ロジスティック回帰の混同行列")
    confusion = result.confusion_counts
    confusion_table = pd.DataFrame(
        [
            [confusion["TN"], confusion["FP"]],
            [confusion["FN"], confusion["TP"]],
        ],
        index=["実際：4着以下（0）", "実際：3着以内（1）"],
        columns=["予測：4着以下（0）", "予測：3着以内（1）"],
    )
    st.dataframe(confusion_table, width="stretch")
    st.markdown(
        f"""
- **TP = {confusion["TP"]}件**：3着以内と分析し、実際も3着以内
- **FP = {confusion["FP"]}件**：3着以内と分析したが、実際は4着以下
- **FN = {confusion["FN"]}件**：4着以下と分析したが、実際は3着以内
- **TN = {confusion["TN"]}件**：4着以下と分析し、実際も4着以下
"""
    )


with st.expander("モデル比較"):
    comparison_columns = ["Model", *METRIC_NAMES]
    if fixed_results is None:
        st.info(
            "保存済みのモデル比較結果を表示できません。"
            f"{MODEL_COMPARISON_PATH.name}を確認してください。"
        )
    else:
        try:
            model_comparison = fixed_results.copy()
            if "Version" in model_comparison.columns:
                model_comparison = model_comparison[
                    model_comparison["Version"].eq("Experimental")
                ].copy()
            missing_columns = sorted(
                set(comparison_columns) - set(model_comparison.columns)
            )
            if missing_columns:
                raise ValueError(
                    "必要な列がありません: " + "、".join(missing_columns)
                )
        except ValueError as error:
            st.warning(f"モデル比較結果を読み込めませんでした: {error}")
        else:
            st.dataframe(
                model_comparison[comparison_columns],
                hide_index=True,
                width="stretch",
                column_config={
                    metric_name: st.column_config.NumberColumn(
                        metric_name, format="%.3f"
                    )
                    for metric_name in METRIC_NAMES
                },
            )
            st.caption(
                "第3回20特徴量版の固定75%/25%テスト評価です。"
                "正式モデルを自動的に切り替える処理はありません。"
            )


with st.expander("時系列交差検証"):
    st.write(
        "過去の学習期間を少しずつ広げ、その直後の未来期間で性能を"
        "5回確認した安定性評価です。固定75%/25%テストとは別の結果です。"
    )
    if not (
        TIME_SERIES_CV_FOLDS_PATH.exists()
        and TIME_SERIES_CV_SUMMARY_PATH.exists()
    ):
        st.info(
            "保存済みの時系列交差検証結果が見つかりません。"
            "resultsフォルダーを確認してください。"
        )
    else:
        summary_required = {
            "Model",
            "Accuracy Mean",
            "Precision Mean",
            "Recall Mean",
            "F1 Mean",
            "F1 Std",
            "ROC-AUC Mean",
            "ROC-AUC Std",
        }
        fold_required = {
            "Model",
            "Fold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Train Start",
            "Train End",
            "Validation Start",
            "Validation End",
            "Train Rows",
            "Validation Rows",
            "Train Races",
            "Validation Races",
            "Validation Top3 Rate",
        }
        try:
            cv_summary = load_saved_result(str(TIME_SERIES_CV_SUMMARY_PATH))
            cv_folds = load_saved_result(str(TIME_SERIES_CV_FOLDS_PATH))
            if "version" in cv_summary.columns:
                cv_summary = cv_summary[
                    cv_summary["version"].eq("Experimental")
                ].copy()
            if "version" in cv_folds.columns:
                cv_folds = cv_folds[
                    cv_folds["version"].eq("Experimental")
                ].copy()
            cv_summary = cv_summary.rename(
                columns={
                    "model": "Model",
                    "accuracy_mean": "Accuracy Mean",
                    "precision_mean": "Precision Mean",
                    "recall_mean": "Recall Mean",
                    "f1_mean": "F1 Mean",
                    "f1_std": "F1 Std",
                    "roc_auc_mean": "ROC-AUC Mean",
                    "roc_auc_std": "ROC-AUC Std",
                }
            )
            cv_folds = cv_folds.rename(
                columns={
                    "model": "Model",
                    "fold": "Fold",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "roc_auc": "ROC-AUC",
                    "train_start_date": "Train Start",
                    "train_end_date": "Train End",
                    "validation_start_date": "Validation Start",
                    "validation_end_date": "Validation End",
                    "train_rows": "Train Rows",
                    "validation_rows": "Validation Rows",
                    "train_races": "Train Races",
                    "validation_races": "Validation Races",
                    "validation_positive_rate": "Validation Top3 Rate",
                }
            )
            missing_summary = sorted(summary_required - set(cv_summary.columns))
            missing_folds = sorted(fold_required - set(cv_folds.columns))
            if missing_summary or missing_folds:
                raise ValueError(
                    "必要な列がありません: "
                    + "、".join(missing_summary + missing_folds)
                )
        except (
            OSError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
            st.warning(f"時系列交差検証結果を読み込めませんでした: {error}")
        else:
            mean_columns = [
                "Model",
                "Accuracy Mean",
                "Precision Mean",
                "Recall Mean",
                "F1 Mean",
                "ROC-AUC Mean",
            ]
            stability_columns = [
                "Model",
                "ROC-AUC Mean",
                "ROC-AUC Std",
                "F1 Mean",
                "F1 Std",
            ]
            number_config = {
                column: st.column_config.NumberColumn(column, format="%.3f")
                for column in set(mean_columns + stability_columns) - {"Model"}
            }
            st.write("#### モデルごとの平均指標")
            st.dataframe(
                cv_summary[mean_columns],
                hide_index=True,
                width="stretch",
                column_config=number_config,
            )
            st.write("#### ROC-AUC・F1の平均と標準偏差")
            st.dataframe(
                cv_summary[stability_columns],
                hide_index=True,
                width="stretch",
                column_config=number_config,
            )
            st.caption(
                "標準偏差が小さいほど、評価時期が変わっても結果が"
                "安定していたと解釈できます。"
            )
            st.write("#### 各foldの詳細結果")
            st.dataframe(
                cv_folds.sort_values(["Model", "Fold"]),
                hide_index=True,
                width="stretch",
                column_config={
                    name: st.column_config.NumberColumn(name, format="%.3f")
                    for name in [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                        "ROC-AUC",
                        "Validation Top3 Rate",
                    ]
                },
            )


with st.expander("特徴量改善実験"):
    experiment_summary = pd.DataFrame(
        [
            {
                "実験": "第1回",
                "内容": "履歴件数・履歴有無",
                "判断": "不採用",
            },
            {"実験": "第2回", "内容": "距離帯成績", "判断": "不採用"},
            {
                "実験": "第3回",
                "内容": "レース条件4特徴量",
                "判断": "採用",
            },
        ]
    )
    st.dataframe(
        experiment_summary,
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "第3回でracecourse、field_size、race_class_name、"
        "relative_horse_numberを正式採用しました。"
    )


st.header("5. 分析結果の解説")
missing_value_count = int(races[FEATURE_COLUMNS].isna().sum().sum())
explanation_context = ExplanationContext(
    race_id=str(selected_race_id),
    predictions=race_predictions,
    metrics=result.metrics,
    feature_names=FEATURE_COLUMNS,
    total_rows=len(races),
    evaluation_rows=len(result.predictions),
    missing_value_count=missing_value_count,
)
local_explanation, local_mode = explain_model_results(explanation_context)
if "gemini_explanations" not in st.session_state:
    st.session_state.gemini_explanations = {}

st.caption(
    "生成AIはPythonモデルが算出した推定確率を説明するだけで、"
    "独自に着順や確率を予測しません。"
)
if st.button("生成AIで分析結果を解説", type="primary"):
    with st.spinner("分析結果を説明しています..."):
        generated = explain_model_results_with_gemini(explanation_context)
    st.session_state.gemini_explanations[str(selected_race_id)] = {
        "text": generated.text,
        "source": generated.source,
        "status_message": generated.status_message,
        "model_name": generated.model_name,
    }

stored_explanation = st.session_state.gemini_explanations.get(
    str(selected_race_id)
)
if stored_explanation is None:
    active_explanation = local_explanation
    st.info("ローカル説明")
    st.caption(local_mode)
else:
    active_explanation = stored_explanation["text"]
    if stored_explanation["source"] == "gemini":
        st.success("Geminiによる分析結果の解説")
        st.caption(
            f"使用モデル：{stored_explanation['model_name']}。"
            "同じレースの単なる画面再描画ではAPIを再呼び出ししません。"
        )
    else:
        st.info("ローカル説明")
        st.caption(stored_explanation["status_message"])
st.markdown(active_explanation)

with st.expander("実際の着順との比較（モデル結果の振り返り）"):
    st.caption(
        "推定確率50%を分類境界として過去の実着順との不一致を整理します。"
        "不一致の理由を特定するものではありません。"
    )
    st.markdown(build_ai_review(explanation_context))


st.header("6. 分析結果をMarkdownで出力")
try:
    markdown_report = generate_markdown_report(
        race_id=str(selected_race_id),
        predictions=race_predictions,
        metrics=result.metrics,
        feature_names=FEATURE_COLUMNS,
        total_rows=len(races),
        evaluation_rows=len(result.predictions),
        missing_value_count=missing_value_count,
        explanation=active_explanation,
    )
except (ValueError, KeyError, TypeError) as error:
    st.warning(f"Markdownレポートを作成できませんでした: {error}")
else:
    st.download_button(
        label="分析結果をMarkdownでダウンロード",
        data=markdown_report,
        file_name=f"race_analysis_{selected_race_id}.md",
        mime="text/markdown",
    )


with st.expander("データ漏洩対策"):
    st.write(
        "正式モデルは第3回で採用した20特徴量だけを使用しています。"
        "説明変数にはレース前の競走条件と対象レースより前の過去成績だけを使用し、"
        "finish_position、target、race_id、race_date、horse_id、horse_name、"
        "jockey_idは使用していません。"
    )
