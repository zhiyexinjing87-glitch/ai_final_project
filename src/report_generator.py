from collections.abc import Mapping, Sequence

import pandas as pd


def _safe_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metrics_table(metrics: Mapping[str, float]) -> str:
    rows = ["| 指標 | 値 |", "|---|---:|"]
    for name in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"):
        rows.append(f"| {name} | {float(metrics.get(name, 0.0)):.3f} |")
    return "\n".join(rows)


def _prediction_table(predictions: pd.DataFrame) -> str:
    rows = [
        "| 馬番 | 馬名 | 推定確率 | 推定クラス | 実着順 | 実際の3着以内 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    ordered = predictions.sort_values("top3_probability", ascending=False)
    for row in ordered.itertuples():
        rows.append(
            f"| {int(row.gate_number)} | {_safe_text(row.horse_name)} | "
            f"{row.top3_probability:.1%} | {int(row.predicted_target)} | "
            f"{int(row.finish_position)} | {int(row.target)} |"
        )
    return "\n".join(rows)


def _comparison(predictions: pd.DataFrame, threshold: float = 0.5) -> str:
    finish = pd.to_numeric(predictions["finish_position"], errors="coerce")
    high_but_missed = predictions[
        (predictions["top3_probability"] >= threshold) & (finish >= 4)
    ].sort_values("top3_probability", ascending=False)
    low_but_placed = predictions[
        (predictions["top3_probability"] < threshold) & (finish <= 3)
    ].sort_values("top3_probability", ascending=False)

    def lines(rows: pd.DataFrame) -> str:
        if rows.empty:
            return "- 該当なし"
        return "\n".join(
            f"- {_safe_text(row.horse_name)}: 推定確率 {row.top3_probability:.1%}、"
            f"実着順 {int(row.finish_position)}着"
            for row in rows.itertuples()
        )

    return (
        "### 高確率だったが実際には4着以下\n\n"
        f"{lines(high_but_missed)}\n\n"
        "### 低確率だったが実際には3着以内\n\n"
        f"{lines(low_but_placed)}"
    )


def _without_headings(markdown: str) -> str:
    body_lines = [
        line for line in markdown.splitlines() if not line.lstrip().startswith("#")
    ]
    return "\n".join(body_lines).strip()


def generate_markdown_report(
    *,
    race_id: str,
    predictions: pd.DataFrame,
    metrics: Mapping[str, float],
    feature_names: Sequence[str],
    total_rows: int,
    evaluation_rows: int,
    missing_value_count: int,
    explanation: str,
) -> str:
    if predictions.empty:
        raise ValueError("レポート対象の推定結果がありません。")

    dates = pd.to_datetime(predictions["race_date"], errors="coerce").dropna()
    race_date = dates.iloc[0].date().isoformat() if not dates.empty else "不明"
    missing_text = (
        f"{missing_value_count}セル（モデル前処理で補完）"
        if missing_value_count
        else "なし"
    )
    feature_list = "\n".join(f"- {_safe_text(name)}" for name in feature_names)

    return f"""# 競走データ分析レポート

## 1. 対象データ

- race_id: {_safe_text(race_id)}
- race_date: {race_date}
- 元データ件数: {total_rows}件
- 評価データ件数: {evaluation_rows}件
- このレースの出走馬数: {len(predictions)}頭
- 説明変数の欠損値: {missing_text}

## 2. 使用した特徴量

{feature_list}

finish_position、target、race_id、race_date、horse_id、horse_name、jockey_id は説明変数に使用していません。

## 3. モデル

クラス重みを調整したロジスティック回帰を使用しています。数値列は欠損値を中央値で補完して標準化し、カテゴリ列は欠損値を最頻値で補完した後、OneHotEncoderで変換しています。学習用と評価用はrace_date順に分割しています。

## 4. 評価指標

{_metrics_table(metrics)}

## 5. 各馬の推定結果

{_prediction_table(predictions)}

## 6. 実際の結果との比較

推定確率50%を分類境界として比較しています。

{_comparison(predictions)}

不一致の理由を事実として特定するものではありません。

## 7. 生成AIによる説明

{_without_headings(explanation)}

この説明はPythonモデルが算出した結果をローカルテンプレートで整理したもので、外部AI APIは使用していません。

## 8. モデルの限界

約2年半の過去データを使った授業・研究用の評価であり、実際の結果を保証するものではありません。モデルが利用していない馬の状態やレース展開などが結果に関係した可能性があります。評価指標はデータ件数や時系列の分割位置によって変動する可能性があります。

## 9. 今後の改善案

- 過去時点で利用可能だった特徴量を追加する
- より多くの過去レースを収集する
- レース日単位の時系列交差検証を行う
- 混同行列と確率校正を確認する
- 同じ分割条件で複数の分類モデルを比較する

対象レース終了後に判明する情報を説明変数へ混ぜない確認が必要です。
"""
