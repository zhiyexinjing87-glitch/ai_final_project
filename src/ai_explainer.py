import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ExplanationContext:
    race_id: str
    predictions: pd.DataFrame
    metrics: Mapping[str, float]
    feature_names: Sequence[str]
    total_rows: int
    evaluation_rows: int
    missing_value_count: int


@dataclass(frozen=True)
class GeneratedExplanation:
    text: str
    source: str
    status_message: str
    model_name: str | None = None


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
MAX_GEMINI_HORSES = 8
PROHIBITED_OUTPUT_TERMS = (
    "馬券",
    "賭け",
    "購入",
    "金額",
    "買い目",
    "オッズ",
    "利益",
    "買うべき",
    "本命",
    "おすすめ",
)


def _metric_text(metrics: Mapping[str, float]) -> str:
    labels = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
    return "、".join(
        f"{label} {float(metrics.get(label, 0.0)):.3f}" for label in labels
    )


def _probability_summary(predictions: pd.DataFrame) -> str:
    ordered = predictions.sort_values("top3_probability", ascending=False)
    descriptions = [
        f"{row.horse_name} {row.top3_probability:.1%}（実着順: {int(row.finish_position)}着）"
        for row in ordered.itertuples()
    ]
    return "、".join(descriptions)


def _feature_tendency(predictions: pd.DataFrame) -> str:
    ordered = predictions.sort_values("top3_probability", ascending=False)
    high_group = ordered.head(min(3, len(ordered)))
    if high_group.empty:
        return "比較対象となる評価行がありません。"

    tendencies: list[str] = []
    for column, label, unit in (
        ("gate_number", "馬番平均", ""),
        ("age", "年齢平均", "歳"),
        ("distance_m", "距離平均", "m"),
        ("carried_weight", "斤量平均", "kg"),
        ("avg_finish_last3", "直近3走平均着順", ""),
        ("top3_rate_last5", "直近5走3着内率", ""),
    ):
        values = pd.to_numeric(high_group[column], errors="coerce")
        if values.notna().any():
            tendencies.append(f"{label} {values.mean():.1f}{unit}")

    conditions = high_group["track_condition"].dropna().astype(str)
    if not conditions.empty:
        tendencies.append("馬場状態 " + "・".join(conditions.unique()))

    return "、".join(tendencies) if tendencies else "比較可能な特徴量がありません。"


def _miss_summary(predictions: pd.DataFrame) -> str:
    false_positive = (
        (predictions["predicted_target"] == 1) & (predictions["target"] == 0)
    ).sum()
    false_negative = (
        (predictions["predicted_target"] == 0) & (predictions["target"] == 1)
    ).sum()
    return f"見逃しが{int(false_negative)}件、過大評価が{int(false_positive)}件"


def _template_explanation(context: ExplanationContext) -> str:
    missing_text = (
        f"{context.missing_value_count}セルあり、前処理で補完"
        if context.missing_value_count
        else "なし"
    )
    features = "、".join(context.feature_names)

    return f"""### 1. 分析結果の要約

評価用レース「{context.race_id}」に対するPythonモデルの出力は、{_probability_summary(context.predictions)}でした。全体の評価指標は{_metric_text(context.metrics)}です。これらは過去データ上の評価結果であり、確実な結果を示すものではありません。

### 2. 高く評価された要因

モデルが使用した特徴量は「{features}」です。推定確率上位の最大3頭には、{_feature_tendency(context.predictions)}という傾向が見られました。ただし、これは上位行の記述的な比較であり、各特徴量が結果を直接引き起こしたとは断定できません。

### 3. 予測が外れた可能性のある理由

この評価レースでは{_miss_summary(context.predictions)}でした。少数の特徴量だけでは、各馬の状態やレース展開など、結果に関係し得る情報を十分に表現できなかった可能性があります。また、ロジスティック回帰の単純な境界では捉えにくい関係が含まれていた可能性もあります。

### 4. データとモデルの限界

元データは{context.total_rows}件、評価対象は{context.evaluation_rows}件で、欠損値は{missing_text}です。限られた期間の過去データであるため、評価指標は分割方法や個々のレースの影響を受ける可能性があります。この説明はモデルが算出した確率・実着順・評価指標だけを整理したものです。

### 5. 次回の改善案

過去レース内で利用可能な特徴量を増やすこと、レース日単位の時系列交差検証を導入すること、確率の校正状況や混同行列を確認することが改善候補です。データを増やしたうえで、複数の分類モデルを同じ評価条件で比較すると、結果の安定性を検討しやすくなります。
"""


def explain_model_results(context: ExplanationContext) -> tuple[str, str]:
    """外部通信せず、モデル結果をローカルテンプレートで説明する。"""
    mode = "ローカル説明（外部APIは呼び出していません）"
    try:
        return _template_explanation(context), mode
    except Exception:
        fallback = """### 分析結果の説明

説明文の生成処理でエラーが発生しました。機械学習の評価指標と推定確率は上の表で確認できます。アプリのほかの機能は継続して利用できます。
"""
        return fallback, mode


def _load_dotenv_if_available() -> None:
    """python-dotenvが導入済みの場合だけ、プロジェクトの.envを読む。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _context_value(row: pd.Series, column: str, default: str = "不明") -> str:
    if column not in row.index or pd.isna(row[column]):
        return default
    value = row[column]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _race_number(race_id: str) -> str:
    value = str(race_id).rsplit("_", maxsplit=1)[-1]
    return str(int(value)) if value.isdigit() else value


def build_gemini_prompt(context: ExplanationContext) -> str:
    """選択レースの最小限の情報だけでGemini向けプロンプトを作る。"""
    if context.predictions.empty:
        raise ValueError("説明対象の推定結果がありません。")

    ordered = context.predictions.sort_values(
        "top3_probability", ascending=False
    )
    first = ordered.iloc[0]
    race_date = pd.to_datetime(
        first.get("race_date"), errors="coerce"
    )
    race_date_text = (
        race_date.date().isoformat() if not pd.isna(race_date) else "不明"
    )
    horse_lines: list[str] = []
    for row in ordered.head(MAX_GEMINI_HORSES).itertuples():
        finish = pd.to_numeric(
            getattr(row, "finish_position", None), errors="coerce"
        )
        finish_text = "不明" if pd.isna(finish) else f"{int(finish)}着"
        horse_lines.append(
            f"- 馬番{int(row.gate_number)} {row.horse_name}: "
            f"推定確率{row.top3_probability:.1%}、実際の着順{finish_text}"
        )

    feature_names = "、".join(context.feature_names)
    metrics = context.metrics
    return f"""あなたは機械学習による過去競走データ分析結果を説明するアシスタントです。

以下はPythonのLogistic Regressionが既に算出した結果です。あなた自身で着順や確率を予測せず、この情報だけを初心者向けの日本語300～600文字程度で説明してください。

【レース情報】
- 開催日: {race_date_text}
- 競馬場: {_context_value(first, "racecourse")}
- レース番号: {_race_number(context.race_id)}R
- 距離: {_context_value(first, "distance_m")}m
- 芝・ダート: {_context_value(first, "surface")}
- 馬場状態: {_context_value(first, "track_condition")}
- レースクラス: {_context_value(first, "race_class_name")}
- 出走頭数: {_context_value(first, "field_size")}頭

【推定確率上位{min(MAX_GEMINI_HORSES, len(ordered))}頭】
{chr(10).join(horse_lines)}

【モデル情報】
- モデル: Logistic Regression
- 特徴量数: 20
- 使用特徴量: {feature_names}
- 固定テストROC-AUC: {float(metrics.get("ROC-AUC", 0.0)):.3f}
- F1: {float(metrics.get("F1", 0.0)):.3f}
- Recall: {float(metrics.get("Recall", 0.0)):.3f}

【説明する内容】
1. モデル結果の要約
2. 入力特徴量の範囲で、推定確率が高くなった可能性のある一般的背景
3. 推定と実際の結果が異なる場合に考えられる一般的要因
4. データとモデルの限界
5. 過去データ分析であり、結果を保証しないこと

金銭・購入・賭け・オッズ・利益に関する助言や、それらを連想させる表現は一切出力しないでください。断定を避け、「可能性があります」「考えられます」などの表現を使ってください。"""


def _create_gemini_client(api_key: str):
    """Google公式SDKを遅延importし、APIクライアントを作る。"""
    from google import genai

    return genai.Client(api_key=api_key)


def _local_result(
    context: ExplanationContext,
    status_message: str,
) -> GeneratedExplanation:
    text, _ = explain_model_results(context)
    return GeneratedExplanation(
        text=text,
        source="local",
        status_message=status_message,
    )


def explain_model_results_with_gemini(
    context: ExplanationContext,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    client_factory: Callable[[str], object] | None = None,
) -> GeneratedExplanation:
    """Geminiで説明し、利用不能時は秘密を出さずローカルへ戻す。"""
    _load_dotenv_if_available()
    resolved_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        if api_key is None
        else api_key.strip()
    )
    resolved_model = (
        model_name
        or os.getenv("GEMINI_MODEL", "").strip()
        or DEFAULT_GEMINI_MODEL
    )
    if not resolved_key:
        return _local_result(
            context,
            "Gemini APIキーが設定されていないため、ローカル説明を表示しています。",
        )

    client = None
    try:
        factory = client_factory or _create_gemini_client
        client = factory(resolved_key)
        response = client.models.generate_content(
            model=resolved_model,
            contents=build_gemini_prompt(context),
            config={
                "temperature": 0.2,
                "max_output_tokens": 800,
            },
        )
        response_text = str(getattr(response, "text", "") or "").strip()
        if not response_text:
            return _local_result(
                context,
                "Geminiの応答が空だったため、ローカル説明を表示しています。",
            )
        if any(term in response_text for term in PROHIBITED_OUTPUT_TERMS):
            return _local_result(
                context,
                "Geminiの応答が授業・研究用途の出力条件を満たさなかったため、"
                "ローカル説明を表示しています。",
            )
        return GeneratedExplanation(
            text=response_text,
            source="gemini",
            status_message="Geminiによる分析結果の解説",
            model_name=resolved_model,
        )
    except ImportError:
        return _local_result(
            context,
            "Gemini SDKを利用できないため、ローカル説明を表示しています。",
        )
    except Exception:
        return _local_result(
            context,
            "Gemini APIを利用できなかったため、ローカル説明を表示しています。",
        )
    finally:
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass


def _horse_list(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "該当する馬はありませんでした。"
    ordered = rows.sort_values("top3_probability", ascending=False)
    return "\n".join(
        f"- {row.horse_name}: 推定確率 {row.top3_probability:.1%}、実着順 {int(row.finish_position)}着"
        for row in ordered.itertuples()
    )


def build_ai_review(context: ExplanationContext, threshold: float = 0.5) -> str:
    """推定確率と過去の実着順の不一致をローカルで振り返る。"""
    try:
        predictions = context.predictions
        high_but_missed = predictions[
            (predictions["top3_probability"] >= threshold)
            & (pd.to_numeric(predictions["finish_position"], errors="coerce") >= 4)
        ]
        low_but_placed = predictions[
            (predictions["top3_probability"] < threshold)
            & (pd.to_numeric(predictions["finish_position"], errors="coerce") <= 3)
        ]

        missing_text = (
            f"元データには説明変数の欠損が{context.missing_value_count}セルあり、"
            "補完値の影響を受けた可能性があります。"
            if context.missing_value_count
            else "説明変数に欠損はありませんが、観測されていない情報は含まれていません。"
        )

        return f"""#### 推定確率が高かったが実際には4着以下だった馬

{_horse_list(high_but_missed)}

#### 推定確率が低かったが実際には3着以内だった馬

{_horse_list(low_but_placed)}

#### モデルが見落とした可能性のある要因

現在のモデルは「{"、".join(context.feature_names)}」だけを使用しています。そのため、過去時点で利用可能だった馬の状態、距離への適性、枠順、騎手や競馬場との相性、直近成績、レース展開に関係する情報を捉えられなかった可能性があります。ここに挙げた内容は不一致の原因を特定したものではなく、今後確認する候補です。

#### 追加するとよい特徴量

- 過去レースから計算した直近成績や着順傾向
- 距離、コース、競馬場、馬場状態別の過去成績
- 馬番、出走頭数、斤量の過去からの変化
- 過去時点で確定していた騎手・馬の組み合わせ成績
- レース間隔や過去の走破タイム

追加する場合も、対象レース終了後に判明する着順などを説明変数へ混ぜない確認が必要です。

#### データ不足に関する注意

元データは{context.total_rows}件、評価データは{context.evaluation_rows}件です。{missing_text} 限られた期間の過去データでは、評価期間によって指標や不一致件数が変わる可能性があります。別期間でも時系列評価を繰り返す必要があります。
"""
    except Exception:
        return """AI反省会の作成中にエラーが発生しました。上の推定確率と実着順の表は引き続き確認できます。"""
