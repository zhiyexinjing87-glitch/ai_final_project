# Race Data Insight

大学授業・研究向けに、約2年半の過去レースデータで二値分類を学ぶStreamlitアプリです。各馬が3着以内だったかを20特徴量版ロジスティック回帰で分類し、時系列で分けた評価データについて推定確率と評価指標を表示します。任意でGoogle Gemini APIを使い、機械学習が算出した結果だけを自然言語で説明できます。

実際の馬券購入、購入額、オッズ、利益、期待値の計算や、現行レースの予想・推奨は行いません。

## 動作環境

- Windows 10／11
- Python 3.12以降
- UTF-8で保存されたCSV

Pythonが未導入の場合は、Python公式インストーラーでインストールし、py --version が成功することを確認してください。

## セットアップ

PowerShellでプロジェクトフォルダへ移動します。

~~~powershell
cd "C:\Users\T0M0Y\OneDrive\デスクトップ\AI開発"
py --version
~~~

既存の .venv が別のPythonを参照して動かない場合は、削除して作り直します。

~~~powershell
Remove-Item -LiteralPath ".venv" -Recurse -Force
py -m venv .venv
~~~

仮想環境を有効化しなくても、次のコマンドで依存関係を導入できます。

~~~powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Gemini説明を利用する場合だけ、`.env.example`をコピーして`.env`を作成します。

~~~powershell
Copy-Item -LiteralPath ".env.example" -Destination ".env"
~~~

`.env`へ自分のキーを設定します。実際のキーはREADMEやソースコードへ書かないでください。

~~~dotenv
GEMINI_API_KEY=ここに自分のAPIキー
GEMINI_MODEL=gemini-3.6-flash
~~~

`GEMINI_MODEL`は任意です。未設定時は`gemini-3.6-flash`を使用します。`.env`がない場合やキーが空の場合もアプリは正常に動作し、説明部分だけローカルテンプレートへ切り替わります。

## アプリの実行

正式モデルファイルがない場合、または正式モデル用データを作り直した場合は、先に次を実行します。

~~~powershell
.\.venv\Scripts\python.exe scripts\train_final_model.py
~~~

この処理は古い75%の89,446行だけで正式Pipelineを学習し、`models/final_logistic_regression.pkl`へ保存します。新しい25%は評価用として学習へ混ぜません。

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

表示されたローカルURLをブラウザで開きます。

## 機械学習と生成AIの役割

### 機械学習

保存済みの20特徴量版Logistic Regressionが、各馬の3着以内となる傾向を数値的な推定確率として計算します。Geminiはこの計算に関与しません。

### 生成AI

利用者が「生成AIで分析結果を解説」ボタンを押した場合だけ、Gemini APIへ選択レースの概要、推定確率上位8頭、実際の過去着順、主要評価指標、使用特徴量名を送ります。Geminiはその結果を初心者向けの文章に整理するだけで、独自に着順や確率を予測しません。APIキー未設定、SDK未導入、認証・通信・モデル名エラー、空応答の場合はローカル説明へ戻ります。

## 最終モデル

- モデル：Logistic Regression
- 特徴量：既存16特徴量＋`racecourse`、`field_size`、`race_class_name`、`relative_horse_number`の計20項目
- データ：約2年半の過去実データ120,273行
- 前処理：数値欠損を中央値補完して標準化、カテゴリ欠損を最頻値補完して`OneHotEncoder(handle_unknown="ignore")`
- 分類器：`class_weight="balanced"`、`max_iter=1500`、既定solver・既定0.5閾値

第3回Experimental Logistic Regressionの記録済み評価は、固定テストでROC-AUC 0.758、F1 0.489、5-fold時系列CVでROC-AUC平均0.740、F1平均0.478です。数値は`results/feature_experiment_3_fixed.csv`と`results/feature_experiment_3_cv_summary.csv`を正とします。

特徴量改善実験の結論は次のとおりです。

| 実験 | 追加内容 | 判断 |
|---|---|---|
| 第1回 | 履歴件数・履歴有無 | 不採用 |
| 第2回 | 距離帯別3着内率 | 不採用 |
| 第3回 | 競馬場・出走頭数・レース名／クラス・相対馬番 | 採用 |

保存モデルはPipeline・前処理・分類器を一体で保存しています。アプリはこのファイルを読み込み、固定25%評価データへ予測するだけで、起動時にモデル比較やCVを実行しません。

本アプリは授業・研究用途の過去データ分析であり、実際の競走結果を保証するものではありません。

## テスト

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests -q
~~~

pytest 単体ではなく python -m pytest を使うことで、PATHの設定に依存せず仮想環境内のpytestを実行できます。

## 分類モデル候補の比較

既存のロジスティック回帰を置き換えず、scikit-learnの一般的な分類モデルを同じ時系列train/test分割で比較します。比較を実行すると、結果が `results/model_comparison.csv` に保存されます。

~~~powershell
.\.venv\Scripts\python.exe scripts\compare_models.py
~~~

Streamlitは保存済みCSVを「モデル候補比較」セクションに表示します。アプリ起動時に全モデルを再学習しないため、データを作り直した場合は比較スクリプトも再実行してください。

## 時系列交差検証

固定75%/25%のtest評価とは別に、過去の学習期間を広げながら直後の未来期間で評価するexpanding-window validationを5foldで実行します。対象はLogistic Regression、Gradient Boosting、AdaBoostの3モデルだけです。

~~~powershell
.\.venv\Scripts\python.exe scripts\time_series_cv.py
~~~

各モデル・各foldの詳細は `results/time_series_cv_folds.csv`、平均・標準偏差・最小・最大は `results/time_series_cv_summary.csv` に保存されます。Streamlit起動時には再学習せず、この2ファイルを「時系列交差検証」セクションへ表示します。

ランダムな交差検証は使用しません。各foldでtrainの最終日より後の日付だけをvalidationとし、同じrace_idを両方へ含めません。欠損値処理、OneHotEncoder、標準化を含むPipelineも各foldのtrainデータだけで学習します。

## 実データCSVの変換

ヘッダーなし・52列・CP932の `sasa.csv` を、既存モデルと同じ列構造のUTF-8 CSVへ変換できます。元の `sasa.csv` は読み取り専用として扱い、上書きしません。

列番号0～51と16特徴量の生成元は、`実データ_列対応表_sasa.xlsx` の「52列対応表」「16特徴量対応」に照合済みです。対応表で確度が低い列やレース後に判明する列は、モデルの説明変数へ使用しません。

~~~powershell
.\.venv\Scripts\python.exe scripts\convert_sasa_to_model_ready.py --input sasa.csv --output model_ready_real_data.csv
~~~

取消・除外等は `race_status_code == 0` かつ `finish_position > 0` を満たさない行として、出力と履歴集計から除外します。履歴特徴量はレース単位で、対象レースより前の有効結果だけから計算します。履歴がない場合は0へ置き換えずNaNのまま保存し、既存Pipelineの欠損補完へ渡します。

`gate_number`には互換性維持のため元データの `horse_number`（馬番）を暫定的に格納します。将来モデルの特徴量名を `horse_number` へ変更する場合は、学習済みモデルや合成データとの互換性を別途検討する必要があります。

## 2年半の実データによる基準評価

`model_ready_real_data_2years.csv`を使い、合成データ版と同じ固定75%/25%時系列分割でBaselineと7モデルを評価します。合成データ版の結果ファイルは上書きしません。

~~~powershell
.\.venv\Scripts\python.exe scripts\compare_real_data_models.py
~~~

比較結果は`results/real_data_model_comparison.csv`、混同行列は`results/real_data_confusion_matrices.csv`、固定分割の詳細は`results/real_data_fixed_split.csv`へ保存します。

固定25%テストを除いた古い75%領域だけで、選定した3モデルの5-fold expanding-window validationを実行します。

~~~powershell
.\.venv\Scripts\python.exe scripts\real_data_time_series_cv.py --models "Logistic Regression" "Gradient Boosting" "AdaBoost"
~~~

各foldは`results/real_data_time_series_cv_folds.csv`、集計は`results/real_data_time_series_cv_summary.csv`へ保存します。固定テスト期間はCVのtrainにもvalidationにも使用しません。

## 特徴量改善実験・第1回

既存16特徴量を変更せず、過去履歴の件数4列と履歴有無3列を追加した23特徴量版を別データで比較します。

~~~powershell
.\.venv\Scripts\python.exe scripts\generate_feature_experiment_1_data.py
.\.venv\Scripts\python.exe scripts\run_feature_experiment_1.py
~~~

実験用データは`model_ready_real_data_exp1.csv`です。追加特徴量の統計は`results/feature_experiment_1_stats.csv`、固定テスト比較は`results/feature_experiment_1_fixed.csv`、同一5foldの詳細と集計は`results/feature_experiment_1_cv_folds.csv`と`results/feature_experiment_1_cv_summary.csv`へ保存します。元のモデル用CSVと基準性能結果は上書きしません。

## 特徴量改善実験・第2回

第1回の追加特徴量は使用せず、既存16特徴量へ`distance_band_top3_rate`（距離帯別3着内率）だけを追加した17特徴量版を比較します。距離帯は実験前に固定し、対象レースより前の同一距離帯の成績だけで値を計算します。

~~~powershell
.\.venv\Scripts\python.exe scripts\generate_feature_experiment_2_data.py
.\.venv\Scripts\python.exe scripts\run_feature_experiment_2.py
~~~

実験用データは`model_ready_real_data_exp2.csv`です。特徴量統計と距離帯分布は`results/feature_experiment_2_stats.csv`と`results/feature_experiment_2_distance_bands.csv`、固定テスト比較は`results/feature_experiment_2_fixed.csv`、同一5foldの詳細と集計は`results/feature_experiment_2_cv_folds.csv`と`results/feature_experiment_2_cv_summary.csv`へ保存します。元の実データ、既存Baseline結果、第1回実験結果は上書きしません。

## 特徴量改善実験・第3回（最終実験）

既存16特徴量へ、レース前に確定する`racecourse`、`field_size`、`race_class_name`、`relative_horse_number`を追加した20特徴量版を比較します。第1回・第2回の追加特徴量は使用しません。

~~~powershell
.\.venv\Scripts\python.exe scripts\generate_feature_experiment_3_data.py
.\.venv\Scripts\python.exe scripts\run_feature_experiment_3.py
~~~

実験用データは`model_ready_real_data_exp3.csv`です。追加特徴量の統計は`results/feature_experiment_3_stats.csv`、カテゴリ件数は`results/feature_experiment_3_category_counts.csv`、固定テスト比較は`results/feature_experiment_3_fixed.csv`、同一5foldの詳細と集計は`results/feature_experiment_3_cv_folds.csv`と`results/feature_experiment_3_cv_summary.csv`へ保存します。この第3回を特徴量改善の最終実験とし、第4回実験は行いません。

## データ

正式アプリは`model_ready_real_data_exp3.csv`を読み込みます。第1回・第2回の追加特徴量は正式モデルに含めません。

data/synthetic_races.csv は過去の教育実験と既存テストを再現するために残している架空データです。既定設定では550レース、6,569行、878頭、80騎手です。

使用する特徴量：

- age
- distance_m
- gate_number
- carried_weight
- horse_weight
- weight_change
- days_since_last
- starts_last_180d
- avg_finish_last3
- top3_rate_last5
- distance_top3_rate
- surface_top3_rate
- jockey_top3_rate
- sex
- surface
- track_condition
- racecourse
- field_size
- race_class_name
- relative_horse_number

使用しない列：

- finish_position
- target
- horse_id
- horse_name
- jockey_id
- race_id
- race_date

数値列の欠損は中央値で補完して標準化します。sex、surface、track_condition、racecourse、race_class_nameは最頻値で補完し、OneHotEncoderで変換します。前処理は学習データだけでfitされます。学習用と評価用は race_date 順に分割し、同じレースを両方へ含めません。

直近成績や3着内率は、そのレースの処理前に存在する過去履歴だけで計算します。レース終了後に着順を履歴へ追加するため、未来結果は使いません。

## 評価上の注意

Accuracy、Precision、Recall、F1、ROC-AUCは、選択中の1レースではなく評価用データ全体について計算します。そのため、表示する評価レースを変更しても指標は変わりません。

seed 42の既定データで確認した指標は、Accuracy 0.736、Precision 0.486、Recall 0.751、F1 0.590、ROC-AUC 0.811です。学習用は4,932行・412レース、評価用は1,637行・138レースです。

これは合成データ上の1回の評価であり、実データへ一般化できる性能を示すものではありません。

## Majority Baselineとの比較

Majority Baselineは、学習データで最も多いtargetを評価データの全件へ予測する単純な基準です。現在の学習データではtarget=0が多数派なので、全件を「4着以下」と予測します。

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority Baseline | 0.747 | 0.000 | 0.000 | 0.000 | 算出なし |
| Logistic Regression | 0.736 | 0.486 | 0.751 | 0.590 | 0.811 |

BaselineはAccuracyだけならロジスティック回帰より高いですが、3着以内を1頭も検出できません。全件へ同じクラスを返し、確率の順位付けをしないためROC-AUCは算出していません。

## 混同行列

ロジスティック回帰の評価1,637行に対する件数は次のとおりです。

- TP 311件：3着以内と予測し、実際も3着以内
- FP 329件：3着以内と予測したが、実際は4着以下
- FN 103件：4着以下と予測したが、実際は3着以内
- TN 894件：4着以下と予測し、実際も4着以下

TP、FP、FN、TNの合計は評価データ件数1,637行と一致します。

## 分析説明とAPIキー

通常表示ではPythonモデルの結果をローカルテンプレートで文章化し、外部通信を行いません。「生成AIで分析結果を解説」ボタンを押した場合だけ、`google-genai`を使ってGemini APIを呼び出します。同じレースの説明は`st.session_state`へ保存され、単なる画面再描画では再送信しません。

APIキーは環境変数`GEMINI_API_KEY`から読み込みます。`.env`は`.gitignore`の対象です。APIエラーの詳細やキーを画面・ログへ出力せず、既存のローカル説明へ切り替えます。実際の`.env`やAPIキーをGit管理・提出ZIPへ含めないでください。

## 開発の流れ

1. 合成データでStreamlitと機械学習のMVPを作成
2. 実データへ移行
3. 約3か月分では履歴特徴量の欠損が大きいことを確認
4. 約2年半へデータ期間を拡張
5. 複数分類モデルを同じ固定テストで比較
6. expanding-window方式の時系列交差検証を実施
7. 特徴量改善実験を3回行い、第3回のレース条件4特徴量を採用
8. 20特徴量版Logistic Regressionを正式統合
9. Geminiによる任意の結果説明とローカルフォールバックを追加

## 主な構成

~~~text
app.py                    Streamlit画面
data/synthetic_races.csv  生成済みの架空の過去レースデータ
model_ready_real_data.csv  sasa.csvから生成した既存モデル互換の実データ
scripts/generate_synthetic_data.py  合成データ生成器
scripts/compare_models.py   モデル候補比較を実行してCSVへ保存
scripts/time_series_cv.py  expanding-window validationを実行してCSVへ保存
scripts/convert_sasa_to_model_ready.py  CP932実データをモデル用CSVへ変換
src/model.py              読込み、前処理、学習、評価
src/model_comparison.py   同一分割での候補モデル学習と評価
src/time_series_cv.py     時系列fold作成、3モデル評価、集計
src/real_data_converter.py  52列の割当て、整形、履歴特徴量生成、検証
src/ai_explainer.py       Gemini説明、ローカル説明、結果の振り返り
src/report_generator.py   Markdownレポート生成
results/model_comparison.csv  保存済みのモデル比較結果
results/time_series_cv_folds.csv  各モデル・各foldの詳細結果
results/time_series_cv_summary.csv  モデル別の平均・標準偏差・最小・最大
tests/test_model.py       入力異常とモデル処理の既存テスト
tests/test_synthetic_data.py  行数、着順、履歴漏洩、時系列分割のテスト
tests/test_evaluation.py  Baseline、混同行列、既存指標不変のテスト
tests/test_model_comparison.py  候補比較の分割、特徴量、指標不変テスト
tests/test_time_series_cv.py  expanding-window分割と集計のテスト
tests/test_real_data_converter.py  実データ変換と未来漏洩防止のテスト
~~~

## 既知の限界

- 約2年半の提供データに限定され、他期間や他データへの一般化は保証されない
- 固定test評価は単一分割（別途5-fold時系列交差検証で安定性を確認）
- `race_class_name`はカテゴリ数が多く、未知カテゴリではその情報を利用できない
- 履歴が少ない馬では過去成績特徴量の欠損が多い
- 確率校正は未実装
- 実データは入力CSV内で確認できる過去履歴だけを利用
- Gemini説明は外部API・モデルの利用可否、通信状況、生成内容に依存
- Geminiを利用しない場合は決定的なローカルテンプレートで説明

これらの制約を踏まえ、本アプリは教育用の評価実験としてのみ使用します。
