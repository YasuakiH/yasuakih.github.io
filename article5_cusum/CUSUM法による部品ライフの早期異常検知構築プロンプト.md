# フリート異常検出ダッシュボード — LLM 編集引き継ぎプロンプト

このドキュメントは、添付する Python スクリプト (`fleet_anomaly_detection.py`) を
LLMを用いて編集・改良する際の引き継ぎ情報です。
このドキュメントをシステムプロンプトまたは最初のユーザーメッセージとして貼り付けてください。

---

## 1. システム概要

**スクリプト名：** `fleet_anomaly_detection.py`
**実行環境：** JupyterLab(ipywidgets によるスライダー操作)
**目的：** 機械フリートにおける定期交換部品への不良品混入を、CUSUM 法で早期に異常検知するシミュレーター

---

## 2. コード全体の構成

```
fleet_anomaly_detection.py
│
├── ユーティリティ関数
│   └── weibull_axis_value()        # 累積故障率→ワイブル軸変換(現状は未使用だが保持)
│
├── モジュール定数
│   ├── CAUSE_PREVENTIVE = 0        # 交換原因コード：予防保守(打ち切り)
│   └── CAUSE_FAILURE    = 1        # 交換原因コード：寿命到達による故障
│
├── class ReliabilitySimulator      # シミュレーション・エンジン
│   ├── __init__(cfg, mix_rate_percent)
│   ├── _new_machine()              # 機械辞書の生成ヘルパー
│   ├── _draw_replacement_part()    # 不良品混入の確率判定ヘルパー
│   └── _run_simulation()           # シミュレーション本体
│       ├── self.df_events          # 交換イベント記録 DataFrame
│       └── self.df_daily           # 日次集計 DataFrame
│
├── class Config                    # 全パラメータの集中管理
│   └── update()                    # ALPHA_GOOD / ALPHA_BAD を逆算
│
├── update_dashboard()              # 7 チャートの描画関数(スライダーから呼ばれる)
│
└── UI(ipywidgets)
    ├── baseline_day_slider         # T1：不良品混入開始日(ワイブル・KM用)
    ├── evaluation_day_slider       # T2：評価判定日
    └── mix_rate_slider             # 不良混入率 [%]
```

---

## 3. 物理モデルの前提

- **良品部品**：摩耗故障モード(ワイブル β = 1.5、時間とともに故障率が増加)
- **不良品部品**：初期故障モード(ワイブル β = 0.8、時間とともに故障率が減少)
- **フリート構成**：T=1 から 1 台稼働、30 日ごとに 1 台ずつ増加(RAMP_UP_INTERVAL)
- **部品交換のトリガー**：寿命到達(故障交換)または定期点検での打ち切り(予防交換)
- **不良品混入**：baseline_day 以降の交換時に mix_rate_fraction の確率で混入
- **ALPHA_GOOD / ALPHA_BAD** は直接設定せず、`Config.update()` が自動逆算する(直接変更しないこと)

---

## 4. DataFrame の構造

### `simulator.df_events`(交換イベント記録)

| カラム名 | 型 | 内容 |
|---|---|---|
| `duration` | float | 部品が使用されたサイクル数(≥1.0) |
| `event` | int | 交換原因(0=予防保守, 1=故障)|
| `is_defective` | bool | 交換された部品が不良品かどうか |
| `day` | int | 交換が発生した経過日数 |

### `simulator.df_daily`(日次集計)

| カラム名 | 型 | 内容 |
|---|---|---|
| `good_operating` | int | その日の良品部品搭載機械の台数 |
| `defective_operating` | int | その日の不良品部品搭載機械の台数 |
| `good_replacements` | int | その日の良品部品の交換件数 |
| `defective_replacements` | int | その日の不良品部品の交換件数 |

---

## 5. ダッシュボードのチャート構成(GridSpec 3×3)

| 位置 | チャート | 主な描画内容 |
|---|---|---|
| `grid[0, 0]` | ① フリート稼働・交換状況 | 稼働台数積み上げ棒 + 累積交換数階段グラフ(twinx) |
| `grid[0, 1]` | ② 故障発生タイミング | 日数×寿命サイクルの散布図(良品・不良品を色分け) |
| `grid[1, 0]` | ③ ワイブル確率紙 | T1・T2 のワイブルプロット比較 |
| `grid[1, 1]` | ④ KM プロット | T1・T2 の Kaplan-Meier 生存曲線比較 |
| `grid[2, :2]` | ⑤ CUSUM 異常検知 | CUSUM 値 + 管理限界 + 故障イベント縦線(twinx) |
| `grid[0, 2]` | ⑥ 故障分布(T1) | 寿命サイクルのヒストグラム(良品・不良品積み上げ) |
| `grid[1, 2]` | ⑥ 故障分布(T2) | 同上(T2 時点) |
| `grid[2, 2]` | ⑦ ハザード関数 | 良品・不良品の h(t) を対数スケールで比較 |

---

## 6. CUSUM ロジックの設計経緯と注意点(重要)

### 基本方針
CUSUM のベースライン(正常時の故障率)の推定に、**不良品混入日(T1)を使わない**。
T1 は現場では未知であり、それを前提にした検知ロジックは実用上意味がないため。

### これまでの試行と結論

#### 試行 A：冒頭固定ウィンドウ方式
- シミュレーション開始から固定日数(例：60 日)のデータのみでベースラインを推定する。
- **問題点**：冒頭期間はフリート台数が少ない(1～2 台)ためサンプル数が不足し、
  `baseline_std` が過小になりやすい → `cusum_threshold` が低くなりすぎ → **誤検知が多発**した。

#### 試行 B：スライディングウィンドウ方式
- T2 から過去に一定期間(例：60 日)遡ったウィンドウでベースラインを推定する。
- フリートが増強された後のデータを使えるためサンプル数は安定する。
- **問題点**：ウィンドウが T1(不良品混入開始日)をまたぐと、不良品の故障データが
  ベースラインに混入する(**ベースライン汚染**)。
  → 管理限界が実態より高く設定されてしまい、**検知遅れ**が目立った。
- **結論：採用せず、試行 A の冒頭固定方式に戻した。今後の課題として残る。**

#### 今後の方針
試行 A の誤検知問題(サンプル不足)を解決しつつ、試行 B のベースライン汚染を避けるには、
以下のようなアプローチが考えられる(未実装)。

- **冒頭固定 + サンプル数に応じた適応的な `cusum_threshold`**：
  フリート台数が少ない冒頭期間は `cusum_threshold` を意図的に高めに設定し、
  台数が増えるに従って正規化する。
- **CUSUM の開始タイミングを遅らせる**：
  フリートが一定台数(例：5 台)に達してから CUSUM の計算を開始する。

---

## 7. 編集時の注意点

1. **`Config.ALPHA_GOOD` / `Config.ALPHA_BAD` は直接変更しないこと。**
   `Config.update()` が BETA や EXPECTED_REPLACEMENTS_PER_YEAR から自動逆算する。
   ワイブルスケールを変えたい場合は `BETA_GOOD`・`BETA_BAD`・`EXPECTED_REPLACEMENTS_PER_YEAR` を変更する。

2. **`np.random.seed(42)` はモジュールレベルに固定されている。**
   再現性が必要な場合はここを変更する。シミュレーターを複数回呼び出す場合、
   シードのリセットが必要かどうか意識すること。

3. **CUSUM チャート(⑤)は `twinx()` で左軸(CUSUM 値)と右軸(故障イベント縦線)を使い分けている。**
   凡例は `ax_cusum.get_legend_handles_labels()` と `ax_cusum_r.get_legend_handles_labels()` を
   統合して左軸側にまとめて表示している。

4. **`Weibull_probability_plot()` は `plt.sca(ax)` でカレント軸を明示してから呼ぶこと。**
   `reliability` ライブラリの仕様上、呼び出し前に `plt.sca(ax_weibull)` が必要。

5. **`update_dashboard()` 内でスライダーの `baseline_day`(T1)は ③④⑥ のみで使用する。**
   ⑤ CUSUM ロジックは T1 を参照しない設計になっている(セクション 6 参照)。

6. **描画カラーは `Config.COLOR_GOOD` / `Config.COLOR_DEFECTIVE` に集約されている。**
   個別のチャートにハードコードしないこと。

---

## 8. 主要パラメータ一覧

| パラメータ | 変数名 | 現在値 | 説明 |
|---|---|---|---|
| 設計寿命 | `TARGET_B10` | 800,000 cycles | 良品 B10 寿命 |
| 日稼働サイクル | `CYCLES_PER_DAY` | 20,000 cycles/日 | 1 台あたり |
| 点検間隔 | `INSPECTION_INTERVAL` | 14 日 | 予防保守の周期 |
| フリート増強間隔 | `RAMP_UP_INTERVAL` | 30 日 | 30 日ごとに +1 台 |
| 総期間 | `TOTAL_DAYS` | 360 日 | シミュレーション期間 |
| 良品 β | `BETA_GOOD` | 1.5 | 摩耗故障モード |
| 不良品 β | `BETA_BAD` | 0.8 | 初期故障モード |
| 年間想定交換数 | `EXPECTED_REPLACEMENTS_PER_YEAR` | 50 件 | ALPHA_GOOD 逆算用 |
| 混入開始日 | `CONTAMINATION_DAY` | 181 日 | シミュレーション上の真値 |
| CUSUM ウィンドウ | `CUSUM_BASELINE_WINDOW_DAYS` | 60 日 | 現在は試行A |
| T1 スライダー | `baseline_day_slider` | 初期値 180 | 不良品混入開始日 |
| T2 スライダー | `evaluation_day_slider` | 初期値 210 | ワイブル・KM・CUSUM・ヒストグラムの評価日 |
| 不良率スライダー | `mix_rate_slider` | 初期値 50% | 0～100%、10% 刻み |
