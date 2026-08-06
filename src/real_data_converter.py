from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from os import PathLike
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.model import FEATURE_COLUMNS, NUMERIC_FEATURES


SASA_COLUMN_COUNT = 52
SASA_COLUMNS = [
    "year_2digit",
    "month",
    "day",
    "meeting_round",
    "racecourse",
    "meeting_day",
    "race_number",
    "race_class_name",
    "race_condition_code",
    "surface",
    "course_code",
    "distance_m",
    "track_condition",
    "horse_name",
    "sex",
    "age",
    "jockey_name",
    "carried_weight",
    "field_size",
    "horse_number",
    "finish_position",
    "finish_position_dup",
    "race_status_code",
    "time_diff",
    "popularity_rank",
    "running_time_sec",
    "running_time_code",
    "unknown_constant_27",
    "corner_1_position",
    "corner_2_position",
    "corner_3_position",
    "corner_4_position",
    "last_3f",
    "horse_weight",
    "trainer_name",
    "trainer_region",
    "prize_money_10k_yen",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "runner_record_id",
    "owner_name",
    "breeder_name",
    "sire",
    "dam",
    "broodmare_sire",
    "coat_color",
    "birth_date_yymmdd",
    "reserved_48",
    "reserved_49",
    "reserved_50",
    "pci_like_index",
]
MODEL_READY_COLUMNS = [
    "race_id",
    "race_date",
    "horse_id",
    "horse_name",
    "jockey_id",
    *FEATURE_COLUMNS,
    "finish_position",
]
HISTORY_FEATURE_COLUMNS = [
    "weight_change",
    "days_since_last",
    "starts_last_180d",
    "avg_finish_last3",
    "top3_rate_last5",
    "distance_top3_rate",
    "surface_top3_rate",
    "jockey_top3_rate",
]
NUMERIC_SOURCE_COLUMNS = [
    "year_2digit",
    "month",
    "day",
    "meeting_round",
    "meeting_day",
    "race_number",
    "distance_m",
    "age",
    "carried_weight",
    "field_size",
    "horse_number",
    "finish_position",
    "race_status_code",
    "horse_weight",
]
TEXT_SOURCE_COLUMNS = [
    "racecourse",
    "surface",
    "track_condition",
    "horse_name",
    "sex",
    "horse_id",
    "jockey_id",
]


class SasaConversionError(ValueError):
    """sasa.csvの変換内容を日本語で示すエラー。"""


@dataclass(frozen=True)
class ConversionSummary:
    input_rows: int
    input_columns: int
    valid_rows: int
    excluded_rows: int
    output_rows: int
    output_columns: int
    missing_counts: dict[str, int]
    race_date_min: str
    race_date_max: str
    horse_count: int
    jockey_count: int
    source_sha256: str | None = None


def load_sasa_csv(path: str | PathLike[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            header=None,
            encoding="cp932",
            dtype="string",
            keep_default_na=False,
        )
    except FileNotFoundError as error:
        raise SasaConversionError(f"入力CSVが見つかりません: {path}") from error
    except pd.errors.EmptyDataError as error:
        raise SasaConversionError("sasa.csvが空です。") from error
    except UnicodeDecodeError as error:
        raise SasaConversionError(
            "sasa.csvをCP932で読み込めませんでした。"
        ) from error
    except (pd.errors.ParserError, OSError) as error:
        raise SasaConversionError(
            f"sasa.csvの読み込みに失敗しました: {error}"
        ) from error


def assign_column_names(raw_data: pd.DataFrame) -> pd.DataFrame:
    if raw_data.shape[1] != SASA_COLUMN_COUNT:
        raise SasaConversionError(
            f"sasa.csvは52列必要ですが、{raw_data.shape[1]}列でした。"
        )
    assigned = raw_data.copy()
    assigned.columns = SASA_COLUMNS
    assigned["_source_row"] = np.arange(len(assigned), dtype=int)
    return assigned


def clean_base_columns(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    for column in TEXT_SOURCE_COLUMNS:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )
    for column in NUMERIC_SOURCE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    required_identifiers = ["racecourse", "horse_id", "jockey_id"]
    missing_identifiers = [
        column
        for column in required_identifiers
        if cleaned[column].isna().any()
    ]
    if missing_identifiers:
        raise SasaConversionError(
            "履歴計算に必要な識別子が欠損しています: "
            + "、".join(missing_identifiers)
        )

    # 0kgは実測馬体重ではないため、履歴なしと区別できるNaNにする。
    cleaned.loc[cleaned["horse_weight"] <= 0, "horse_weight"] = np.nan
    return cleaned


def create_race_date(data: pd.DataFrame) -> pd.DataFrame:
    dated = data.copy()
    required = [
        "year_2digit",
        "month",
        "day",
        "meeting_round",
        "meeting_day",
        "race_number",
    ]
    if dated[required].isna().any().any():
        raise SasaConversionError(
            "race_dateまたはrace_idの生成に必要な数値が欠損しています。"
        )
    if not dated["year_2digit"].between(0, 99).all():
        raise SasaConversionError("year_2digitは0～99で指定してください。")

    date_parts = pd.DataFrame(
        {
            "year": 2000 + dated["year_2digit"],
            "month": dated["month"],
            "day": dated["day"],
        },
        index=dated.index,
    )
    dated["race_date"] = pd.to_datetime(date_parts, errors="coerce")
    if dated["race_date"].isna().any():
        invalid_rows = (dated.index[dated["race_date"].isna()] + 1).tolist()
        raise SasaConversionError(
            "日付として解釈できない行があります: "
            + "、".join(map(str, invalid_rows[:10]))
        )
    return dated


def _normalize_racecourse(value: object) -> str:
    return re.sub(r"\s+", "", str(value))


def create_direct_features(data: pd.DataFrame) -> pd.DataFrame:
    featured = data.copy()
    featured["sex"] = featured["sex"].replace(
        {"セ": "せん", "セン": "せん", "せん": "せん"}
    )
    featured["surface"] = featured["surface"].replace(
        {"ダ": "ダート", "ダート": "ダート", "芝": "芝"}
    )
    featured["track_condition"] = featured["track_condition"].replace(
        {"稍": "稍重", "稍重": "稍重", "不": "不良", "不良": "不良"}
    )

    # 実データではhorse_number（馬番）であり、将来的には特徴量名を
    # horse_numberへ変更することを推奨する。現行モデルとの互換性のため暫定利用。
    featured["gate_number"] = featured["horse_number"]

    race_ids = []
    for row in featured.itertuples(index=False):
        race_ids.append(
            f"R{row.race_date:%Y%m%d}_"
            f"{_normalize_racecourse(row.racecourse)}_"
            f"{int(row.meeting_round):02d}_"
            f"{int(row.meeting_day):02d}_"
            f"{int(row.race_number):02d}"
        )
    featured["race_id"] = race_ids
    return featured


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _top3_rate_or_nan(finishes: list[int]) -> float:
    if not finishes:
        return np.nan
    return float(np.mean([finish <= 3 for finish in finishes]))


def _is_valid_result(row: object) -> bool:
    return bool(
        pd.notna(row.race_status_code)
        and float(row.race_status_code) == 0
        and pd.notna(row.finish_position)
        and float(row.finish_position) > 0
    )


def create_history_features(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(
        [
            "race_date",
            "race_number",
            "racecourse",
            "meeting_round",
            "meeting_day",
            "race_id",
            "gate_number",
            "_source_row",
        ],
        kind="stable",
    ).copy()
    duplicate_horses = ordered.duplicated(["race_id", "horse_id"], keep=False)
    if duplicate_horses.any():
        raise SasaConversionError(
            "同じrace_idに同じhorse_idが重複しています。"
        )

    for column in HISTORY_FEATURE_COLUMNS:
        ordered[column] = np.nan

    horse_histories: dict[str, list[dict[str, object]]] = defaultdict(list)
    jockey_histories: dict[str, list[int]] = defaultdict(list)

    # 同じ日・同じrace_numberは競馬場が異なると厳密な開催順を断定できない。
    # その時間帯の全レースを計算してから結果を追加することで、対象レース自身や
    # 開催順が曖昧な他場同番号レースのfinish_positionを混入させない。
    for _, time_rows in ordered.groupby(
        ["race_date", "race_number"], sort=False
    ):
        for index, row in time_rows.iterrows():
            horse_history = horse_histories[str(row["horse_id"])]
            jockey_history = jockey_histories[str(row["jockey_id"])]
            previous = horse_history[-1] if horse_history else None

            current_weight = row["horse_weight"]
            previous_weight = previous["horse_weight"] if previous else np.nan
            ordered.at[index, "weight_change"] = (
                float(current_weight) - float(previous_weight)
                if pd.notna(current_weight) and pd.notna(previous_weight)
                else np.nan
            )
            ordered.at[index, "days_since_last"] = (
                float((row["race_date"] - previous["race_date"]).days)
                if previous is not None
                else np.nan
            )
            ordered.at[index, "starts_last_180d"] = float(
                sum(
                    0 < (row["race_date"] - record["race_date"]).days <= 180
                    for record in horse_history
                )
            )

            finishes = [
                int(record["finish_position"]) for record in horse_history
            ]
            ordered.at[index, "avg_finish_last3"] = _mean_or_nan(
                [float(value) for value in finishes[-3:]]
            )
            ordered.at[index, "top3_rate_last5"] = _top3_rate_or_nan(
                finishes[-5:]
            )
            ordered.at[index, "distance_top3_rate"] = _top3_rate_or_nan(
                [
                    int(record["finish_position"])
                    for record in horse_history
                    if record["distance_m"] == row["distance_m"]
                ]
            )
            ordered.at[index, "surface_top3_rate"] = _top3_rate_or_nan(
                [
                    int(record["finish_position"])
                    for record in horse_history
                    if record["surface"] == row["surface"]
                ]
            )
            ordered.at[index, "jockey_top3_rate"] = _top3_rate_or_nan(
                jockey_history
            )

        for row in time_rows.itertuples(index=False):
            if not _is_valid_result(row):
                continue
            horse_histories[str(row.horse_id)].append(
                {
                    "race_date": row.race_date,
                    "distance_m": row.distance_m,
                    "surface": row.surface,
                    "finish_position": int(row.finish_position),
                    "horse_weight": row.horse_weight,
                }
            )
            jockey_histories[str(row.jockey_id)].append(
                int(row.finish_position)
            )

    return ordered


def build_model_ready_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    valid_mask = (
        data["race_status_code"].eq(0)
        & data["finish_position"].notna()
        & data["finish_position"].gt(0)
    )
    model_ready = data.loc[valid_mask, MODEL_READY_COLUMNS].copy()
    model_ready["finish_position"] = model_ready["finish_position"].astype(int)
    return model_ready.sort_values(
        ["race_date", "race_id", "gate_number"], kind="stable"
    ).reset_index(drop=True)


def validate_output(data: pd.DataFrame) -> None:
    if list(data.columns) != MODEL_READY_COLUMNS:
        raise SasaConversionError(
            "出力列が既存モデル用の列順と一致していません。"
        )
    if data.empty:
        raise SasaConversionError("有効なレース結果が1行もありません。")
    required_values = [
        "race_id",
        "race_date",
        "horse_id",
        "horse_name",
        "jockey_id",
        "finish_position",
    ]
    if data[required_values].isna().any().any():
        raise SasaConversionError("出力の補助列に欠損があります。")
    if data.duplicated(["race_id", "horse_id"]).any():
        raise SasaConversionError("出力にrace_id・horse_idの重複があります。")
    numeric_values = data[NUMERIC_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    if np.isinf(numeric_values.to_numpy(dtype=float)).any():
        raise SasaConversionError("出力特徴量にinfまたは-infがあります。")


def _build_summary(
    raw_data: pd.DataFrame,
    output_data: pd.DataFrame,
    source_hash: str | None,
) -> ConversionSummary:
    valid_rows = len(output_data)
    return ConversionSummary(
        input_rows=len(raw_data),
        input_columns=raw_data.shape[1],
        valid_rows=valid_rows,
        excluded_rows=len(raw_data) - valid_rows,
        output_rows=len(output_data),
        output_columns=output_data.shape[1],
        missing_counts={
            column: int(output_data[column].isna().sum())
            for column in FEATURE_COLUMNS
        },
        race_date_min=output_data["race_date"].min().date().isoformat(),
        race_date_max=output_data["race_date"].max().date().isoformat(),
        horse_count=int(output_data["horse_id"].nunique()),
        jockey_count=int(output_data["jockey_id"].nunique()),
        source_sha256=source_hash,
    )


def transform_sasa_dataframe(
    raw_data: pd.DataFrame,
    source_hash: str | None = None,
) -> tuple[pd.DataFrame, ConversionSummary]:
    assigned = assign_column_names(raw_data)
    cleaned = clean_base_columns(assigned)
    dated = create_race_date(cleaned)
    direct = create_direct_features(dated)
    historical = create_history_features(direct)
    output = build_model_ready_dataframe(historical)
    validate_output(output)
    summary = _build_summary(raw_data, output, source_hash)
    return output, summary


def save_model_ready_csv(
    data: pd.DataFrame,
    output_path: str | PathLike[str],
    input_path: str | PathLike[str] | None = None,
) -> None:
    output = Path(output_path)
    if input_path is not None and output.resolve() == Path(input_path).resolve():
        raise SasaConversionError("元のsasa.csvを出力先には指定できません。")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        output,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d",
    )


def convert_sasa_file(
    input_path: str | PathLike[str],
    output_path: str | PathLike[str],
) -> tuple[pd.DataFrame, ConversionSummary]:
    input_file = Path(input_path)
    if not input_file.exists():
        raise SasaConversionError(f"入力CSVが見つかりません: {input_file}")
    source_hash_before = sha256(input_file.read_bytes()).hexdigest()
    raw_data = load_sasa_csv(input_file)
    output, summary = transform_sasa_dataframe(
        raw_data, source_hash=source_hash_before
    )
    save_model_ready_csv(output, output_path, input_path=input_file)
    source_hash_after = sha256(input_file.read_bytes()).hexdigest()
    if source_hash_before != source_hash_after:
        raise SasaConversionError("元のsasa.csvが処理中に変更されました。")
    return output, summary
