"""
fleet_reliability_simulator.py
================================

フリート早期同等性検証ダッシュボード
Fleet Early Equivalence Testing Dashboard

■シミュレーター作成の背景課題

    新型機を導入したばかりの現場では、稼働台数が少なく故障データが極めて乏しい。
    しかし品質管理を志す担当者は早期に「新型機の部品ライフは旧型機と同等以上か」を
    確認しなければならない。

    問いの本質:
        「何ヶ月のデータを集め、何件の故障を観測すれば、客観的な同等性判断が下せるか?」

    技術的障壁:
        - 予防保守 (PM) によって部品は故障前に交換される。
          この打ち切りデータ (Censored) の発生によって、真の摩耗故障寿命が見えなくなる。

        - 高信頼性の機械では部品故障が起こりにくい。
          時には少数の故障データと打ち切りデータをもとに、部品の信頼性評価が求められる。

    このシミュレーターは上記の問いに答えるため、現場のワイブル分析を足場にして
    「バスタブ型モデルで生成 → ブートストラップでB10ライフ推定 → ワイブルプロットで確認」
    という一貫したストーリーで構築したものである。

■4つの設計思想

    1. Weibull モデルを共通言語とする
       混合ワイブル分布でバスタブ型の部品寿命を生成し (③) 、
       KM推定点をWeibull確率紙変換して β を推定し B10 を逆算し (⑦⑧) 、
       最終的にワイブルプロットで結果を確認する (⑨) 。
       Weibull がデータ生成・推定・可視化を貫く共通軸である。

    2. Censored データを捨てない - 生存データを扱える KM 法の採用
       予防保守による打ち切りデータを除外してはならない。
       Kaplan-Meier 法で Censored を生存率推定の分母に組み込み、生存部品の
       ライフ情報を生かしつつ、バイアスなく Weibull 回帰につなげる。

    3. 点推定でなく不確実性を可視化する
       限られたデータからの点推定は危険。ブートストラップ法による
       B10 ライフ分布化で「現時点での判断の確信度」を定量的に示す。

    4. 統計的根拠で同等性を判定する
       B10 差の 90% 信頼区間が許容下限を上回るかで同等性を判定する
       (TOST; Two One-Sided Tests)。直感的な確率表示も表示。

■コード保守上の留意事項

    - bootstrap_b10_local() 内のガード処理を壊さないこと。
      (β<0.3 排除・exp オーバーフロー防止・最低 Failed 3件チェック)
    - GridSpec の 3×3 レイアウト (①～⑨) はストーリー順に配置されている。
      安易な変更はストーリーの流れを崩すため、必ず main() の docstring を
      確認した上で修正すること。
    - WEIBULL_MODES の確率混合パラメータはデータ生成とバスタブ曲線描画の
      両方で共用されるため、変更時は両方への影響を確認すること。

■プログラミング言語およびライブラリ

    Python-3.10 以降
    pip install numpy pandas matplotlib scipy

■ライセンス

    CC0 (Creative Commons Zero)
"""

# ==============================================================================
# ライブラリインポート
# ==============================================================================
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec

# ==============================================================================
# グローバル定数
# ==============================================================================

# --- フォント設定 (実行環境に合わせて適宜変更) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Arial']

# --- 評価期間 ---
# 新型機の評価月数。値を変えることでフリート導入初期～成熟期の各フェーズをシミュレート可能。
EVAL_MONTHS_NEW: int = 12
# EVAL_MONTHS_NEW = 24
# EVAL_MONTHS_NEW = 36
# EVAL_MONTHS_NEW = 48
# EVAL_MONTHS_NEW = 60
# EVAL_MONTHS_NEW = 72
EVAL_MONTHS_OLD: int = 72   # 旧型機の評価期間 (十分な実績データを想定)

# --- 予防保守設定 ---
# 予防保守(PM)実施サイクル = B10設計ライフ × CENSORING_FACTOR
# 値が大きいほど PM 頻度が下がり、摩耗故障領域のデータが得やすくなる。
CENSORING_FACTOR: float = 4.0
# CENSORING_FACTOR = 8.0

# --- 部品・機械の基本スペック ---
CYCLES_PER_DAY: int = 4000   # 1日あたりの部品稼働サイクル数

B10_DESIGN_NEW: int = 100000  # 新型機の B10 設計ライフ [サイクル]
B10_DESIGN_OLD: int = 80000   # 旧型機の B10 設計ライフ [サイクル]

PM_CYCLES_NEW: float = B10_DESIGN_NEW * CENSORING_FACTOR  # 新型機の予防保守目標サイクル
PM_CYCLES_OLD: float = B10_DESIGN_OLD * CENSORING_FACTOR  # 旧型機の予防保守目標サイクル

# --- 混合ワイブルモデルの故障モード定義 ---
# シミュレーションデータ生成と③バスタブ曲線描画の両方で参照する。
#   beta : ワイブル形状パラメータ β  (β<1: 初期故障, β=1: 偶発故障, β>1: 摩耗故障)
#   prob : 各故障モードの発生確率 (合計 = 1.0)
# ※ スケールパラメータ η は持たない。B10 設計ライフからスケーリングするため不要。
WEIBULL_MODES: list = [
    {'beta': 0.7, 'prob': 0.3},  # 初期故障モード (β < 1)
    {'beta': 1.0, 'prob': 0.4},  # 偶発故障モード (β = 1)
    {'beta': 2.5, 'prob': 0.3},  # 摩耗故障モード (β > 1)
]

# --- ステータス文字列定数 (DataFrame の 'Status' 列で使用) ---
STATUS_FAILED   = 'Failed (事後保守)'
STATUS_CENSORED = 'Censored (予防保守)'
STATUS_ACTIVE   = 'Active (稼働中)'

# --- 同等性検証マージン ---
# B10 ライフ差の許容限界を旧型 B10 に対する比率で定義する。
#   下限 (-10%): 新型が旧型より最大 10% 短くても「同等」とみなす実務的な許容値。
#   上限 (+50%): 技術的な上限 (参考表示のみ、判定には使用しない)。
EQUIV_MARGIN_LOWER_RATIO: float = -0.10
EQUIV_MARGIN_UPPER_RATIO: float = +0.50

# --- ブートストラップ設定 ---
N_BOOTSTRAP_MAIN: int   = 2000  # ⑦同等性検証・⑧サチレーションで使うブートストラップ反復数
N_BOOTSTRAP_OLD:  int   = 2000  # 旧型 B10 基準値算出のブートストラップ反復数
LOCAL_FIT_F_LIMIT: float = 0.3  # 局所フィット対象とする累積故障率の上限 (初期故障領域)

# --- 描画色 ---
COLOR_OLD = 'orange'    # 旧型機を表す色
COLOR_NEW = 'darkblue'  # 新型機を表す色


# ==============================================================================
# ユーティリティ関数
# ==============================================================================

def fmt_kcycles(x: float, _pos) -> str:
    """matplotlib 軸ラベル用フォーマッタ: サイクル数を 'Xk' 形式で表示する。

    Args:
        x:    サイクル数 (軸の値)。
        _pos: matplotlib が内部で渡す位置引数 (未使用)。

    Returns:
        '100k' のような文字列。
    """
    return f'{int(x / 1000)}k'


def weibull_plot_y(failure_rate_percent):
    """累積故障率 [%] をワイブル確率紙の Y 軸値 ln(-ln(1-F)) に変換する。

    Args:
        failure_rate_percent: 累積故障率 [%]。スカラーまたは NumPy 配列。

    Returns:
        ワイブル Y 軸値 ln(-ln(1 - F))。
    """
    return np.log(-np.log(1 - failure_rate_percent / 100))


# ==============================================================================
# シミュレーション・エンジン
# ==============================================================================

def generate_fleet_data(is_new: bool = False, seed: int = 42) -> pd.DataFrame:
    """混合ワイブルモデルによるフリート運用データを生成する。

    ■生成ロジック
        1. WEIBULL_MODES に基づく混合ワイブル分布から部品寿命プールを生成する。
        2. プール全体の B10 が指定の設計ライフに一致するようスケーリングする。
        3. 機械は毎月 1台ずつ導入され、評価期間終了まで稼働する。
        4. 各部品についてその交換理由を「故障 (Failed)」か「予防保守による打ち切り (Censored)」かを
           先に起きたイベントで判定する。そうでなく稼働中であれば「Active」とする。

    Args:
        is_new: True = 新型機, False = 旧型機。この値によって B10目標・PM目標を切り替える。
        seed:   乱数シード (再現性担保)。

    Returns:
        各部品交換イベントを 1 行とする DataFrame。主要列:
            Machine_ID      : 機械 ID
            Status          : STATUS_FAILED / STATUS_CENSORED / STATUS_ACTIVE
            Observed_Cycles : 観測された稼働サイクル数
            Observed_Months : 観測された稼働月数
            Event_Day       : イベント発生日 (シミュレーション開始からの累積日)
            Event_Month     : イベント発生月 (整数)
            Month_Bin       : 集計用月ビン
    """
    np.random.seed(seed)

    # パラメータを新旧で切り替え
    if is_new:
        total_months = EVAL_MONTHS_NEW
        n_machines   = EVAL_MONTHS_NEW   # 毎月 1台導入 → 台数 = 評価月数
        b10_design   = B10_DESIGN_NEW
        pm_cycles    = PM_CYCLES_NEW
    else:
        total_months = EVAL_MONTHS_OLD
        n_machines   = EVAL_MONTHS_OLD
        b10_design   = B10_DESIGN_OLD
        pm_cycles    = PM_CYCLES_OLD

    # -------------------------------------------------------
    # 混合ワイブル分布プールの生成
    # -------------------------------------------------------
    POOL_SIZE = 15000  # シミュレーション中に消費される部品数の上限

    mode_indices = np.random.choice(
        len(WEIBULL_MODES),
        size=POOL_SIZE,
        p=[m['prob'] for m in WEIBULL_MODES]
    )
    raw_lives = np.zeros(POOL_SIZE)
    for i, mode in enumerate(WEIBULL_MODES):
        mask   = (mode_indices == i)
        n_mode = int(np.sum(mask))
        if n_mode > 0:
            # scale=1 の標準ワイブル分布からサンプリング
            raw_lives[mask] = np.random.weibull(mode['beta'], n_mode)

    # 混合分布全体の B10 を指定の設計ライフに合わせてスケーリング
    raw_b10    = np.percentile(raw_lives, 10)
    pool_lives = raw_lives * (b10_design / raw_b10)

    # -------------------------------------------------------
    # フリート運用シミュレーション
    # -------------------------------------------------------
    records  = []
    pool_idx = 0

    for machine_id in range(1, n_machines + 1):
        start_day   = (machine_id - 1) * 30  # 毎月 30 日間隔で導入
        end_day     = total_months * 30
        current_day = start_day

        while current_day < end_day:
            intrinsic_cycles = pool_lives[pool_idx]
            pool_idx += 1

            # 予防保守実施サイクル: 目標値に ±5% のバラツキを付与
            actual_pm_cycles = np.random.normal(pm_cycles, pm_cycles * 0.05)

            # 故障 or 予防保守 の早い方を観測値とする
            observed_cycles = min(intrinsic_cycles, actual_pm_cycles)
            observed_days   = observed_cycles / CYCLES_PER_DAY
            status = STATUS_FAILED if intrinsic_cycles < actual_pm_cycles else STATUS_CENSORED

            if current_day + observed_days >= end_day:
                # その部品が評価期間終了において稼働中 → Active
                remaining_days   = end_day - current_day
                remaining_cycles = remaining_days * CYCLES_PER_DAY
                records.append({
                    'Machine_ID':      machine_id,
                    'Status':          STATUS_ACTIVE,
                    'Observed_Cycles': remaining_cycles,
                    'Observed_Months': remaining_days / 30,
                    'Event_Day':       end_day,
                    'Event_Month':     total_months,
                })
                break
            else:
                current_day += observed_days
                records.append({
                    'Machine_ID':      machine_id,
                    'Status':          status,
                    'Observed_Cycles': observed_cycles,
                    'Observed_Months': observed_days / 30,
                    'Event_Day':       current_day,
                    'Event_Month':     int(current_day // 30) + 1,
                })

    df = pd.DataFrame(records)
    df['Month_Bin'] = (df['Event_Day'] // 30).astype(int) + 1
    return df


# ==============================================================================
# 信頼性統計関数 (Kaplan-Meier / Weibull フィット)
# ==============================================================================

def compute_km_simple(durations: np.ndarray, events: np.ndarray):
    """ブートストラップ内部用の軽量 Kaplan-Meier 推定関数。

    信頼区間は計算しない (速度優先)。ブートストラップ反復処理内での使用を想定。

    Args:
        durations: 各観測の稼働サイクル数配列。
        events:    故障フラグ (True = Failed) のブール配列。

    Returns:
        (times, survival): ユニーク時刻配列と対応する KM 生存確率配列。
    """
    times     = np.sort(np.unique(durations))
    survival  = []
    n_at_risk = len(durations)
    surv_prob = 1.0

    for t in times:
        n_deaths   = int(np.sum((durations == t) & events))
        n_censored = int(np.sum((durations == t) & ~events))
        if n_at_risk > 0:
            surv_prob *= (1 - n_deaths / n_at_risk)
        survival.append(surv_prob)
        n_at_risk -= (n_deaths + n_censored)

    return np.array(times), np.array(survival)


def compute_km_with_ci(durations: np.ndarray, events: np.ndarray):
    """Greenwood の公式による 95% 信頼区間付き Kaplan-Meier 推定。

    グラフ⑥ (KM 曲線) の描画に使用する。

    Args:
        durations: 各観測の稼働サイクル数配列。
        events:    故障フラグ (True = Failed) のブール配列。

    Returns:
        (times, survival, ci_lower, ci_upper):
            times    : ユニーク時刻配列
            survival : KM 生存確率リスト
            ci_lower : 95% CI 下限リスト
            ci_upper : 95% CI 上限リスト
    """
    times         = np.sort(np.unique(durations))
    survival      = []
    ci_lower      = []
    ci_upper      = []
    n_at_risk     = len(durations)
    surv_prob     = 1.0
    greenwood_sum = 0.0  # Greenwood の公式の累積分散項

    for t in times:
        n_deaths   = int(np.sum((durations == t) & events))
        n_censored = int(np.sum((durations == t) & ~events))

        if n_at_risk > 0:
            surv_prob *= (1 - n_deaths / n_at_risk)
            if n_at_risk > n_deaths:
                greenwood_sum += n_deaths / (n_at_risk * (n_at_risk - n_deaths))

        survival.append(float(surv_prob))
        se = surv_prob * np.sqrt(greenwood_sum)
        ci_lower.append(max(0.0, surv_prob - 1.96 * se))
        ci_upper.append(min(1.0, surv_prob + 1.96 * se))
        n_at_risk -= (n_deaths + n_censored)

    return times, survival, ci_lower, ci_upper


def fit_weibull_from_km(km_times: np.ndarray, km_survival: np.ndarray):
    """KM 生存曲線からワイブルパラメータ (β, η) を最小二乗法で推定する。

    ワイブル確率紙変換: X = ln(t), Y = ln(-ln(S(t))) として線形回帰を行う。

    Args:
        km_times:    KM 推定の時刻配列。
        km_survival: KM 推定の生存確率配列。

    Returns:
        (beta, eta, X, Y):
            beta : 形状パラメータ β (回帰の傾き)
            eta  : スケールパラメータ η
            X    : ワイブル X 軸値 ln(t)
            Y    : ワイブル Y 軸値 ln(-ln(S))
            推定不能の場合は (None, None, None, None)。
    """
    km_times    = np.asarray(km_times)
    km_survival = np.asarray(km_survival)

    mask = (km_survival < 1.0) & (km_survival > 0.0) & (km_times > 0)
    if not np.any(mask) or int(np.sum(mask)) < 2:
        return None, None, None, None

    X = np.log(km_times[mask])
    Y = np.log(-np.log(km_survival[mask]))

    slope, intercept = np.polyfit(X, Y, 1)
    beta = slope
    eta  = np.exp(-intercept / beta)
    return beta, eta, X, Y


def fit_weibull_simple(durations: np.ndarray, events: np.ndarray):
    """全データを使った全域ワイブル回帰 (β, η の点推定)。

    局所フィットが不可能な初期段階 (⑧サチレーションチャートの補完) で使用する。

    Args:
        durations: 稼働サイクル数配列。
        events:    故障フラグ (True = Failed) のブール配列。

    Returns:
        (beta, eta): 推定不能な場合は (None, None)。
    """
    times, survival = compute_km_simple(durations, events)
    mask = (survival < 1.0) & (survival > 0.0) & (times > 0)
    if int(np.sum(mask)) < 2:
        return None, None

    X = np.log(times[mask])
    Y = np.log(-np.log(survival[mask]))
    slope, intercept = np.polyfit(X, Y, 1)
    eta = np.exp(-intercept / slope)
    return slope, eta


def bootstrap_b10_local(df: pd.DataFrame,
                        n_boot: int = 200,
                        f_upper_limit: float = 0.3) -> np.ndarray:
    """KM → Weibull確率紙変換 → 線形回帰によるB10ライフのブートストラップ推定。

    ■このシミュレーターの中心となる関数
        KM 法と Weibull 回帰を組み合わせることで、Censored データが多い
        初期段階でも B10 ライフを推定できる。処理の流れは以下のとおり:

        ① 復元抽出でブートストラップサンプルを生成する。
        ② KM 法で生存曲線を推定する。
           → Censored を適切に取り扱って Weibull 回帰につなげる。
        ③ KM 推定点を Weibull 確率紙に変換する。
           X = ln(t),  Y = ln(-ln(S(t)))
           → Weibull 分布に従うなら (X, Y) は直線に乗る。
              直線の傾きが形状パラメータ β の推定値となる。
        ④ F ≤ f_upper_limit の初期故障領域のみで線形回帰を行う。
           → 摩耗故障領域の影響を除いた局所フィットで B10 を外挿する。
        ⑤ B10 に対応する Y 値 = ln(-ln(0.9)) を回帰直線に代入して B10 を逆算する。
        ⑥ n_boot 回反復して B10 の分布を得る (不確実性の可視化)。

    ■数値安定性のためのガード処理 (変更禁止)
        - Failed 件数が 3 未満の場合は空配列を返す。
        - β < 0.3 は B10 が無限大に発散するため採用しない。
        - ln(B10) が 20 を超える場合 (約4.8億サイクル) はオーバーフローとして除外。

    Args:
        df:            対象の fleet データ DataFrame。
        n_boot:        ブートストラップ反復数。
        f_upper_limit: 局所フィット対象の累積故障率上限 (デフォルト 0.3 = 30%)。

    Returns:
        B10 推定値の NumPy 配列。データ不足時は空配列。
    """
    durations = df['Observed_Cycles'].values
    events    = (df['Status'] == STATUS_FAILED).values
    n_samples = len(durations)

    # ガード: Failed が 3 件未満では信頼できる回帰が成立しない
    if int(np.sum(events)) < 3:
        return np.array([])

    b10_estimates = []

    for _ in range(n_boot):
        # 復元抽出
        idx    = np.random.choice(n_samples, n_samples, replace=True)
        d_boot = durations[idx]
        e_boot = events[idx]

        times, survival = compute_km_simple(d_boot, e_boot)
        cum_failure     = 1 - survival
        mask = (cum_failure > 0) & (cum_failure <= f_upper_limit) & (survival > 0)

        if int(np.sum(mask)) < 2:
            continue

        X = np.log(times[mask])
        Y = np.log(-np.log(1 - cum_failure[mask]))

        # ガード: X の重複 (特異行列) を排除
        if len(np.unique(X)) < 2:
            continue

        slope, intercept = np.polyfit(X, Y, 1)

        # ガード: β が小さすぎると B10 が発散する
        if slope <= 0.3:
            continue

        b10_y = np.log(-np.log(0.9))   # F(t) = 10% に対応する Y 値
        b10_x = (b10_y - intercept) / slope

        # ガード: オーバーフロー防止 (exp(20) ≈ 4.8 億サイクル)
        if b10_x >= 20:
            continue
        val = np.exp(b10_x)
        if not np.isinf(val):
            b10_estimates.append(val)

    return np.array(b10_estimates)


# ==============================================================================
# グラフ描画関数 (ax1～ax9)
# ==============================================================================

def plot_ax1_old_fleet_history(ax, df_old: pd.DataFrame) -> None:
    """① 旧型機の現場実績 (累積部品交換数) を積み上げ棒グラフで描画する。

    Failed / Censored を色分けして月次累積件数を可視化する。
    Active は「まだ交換が起きていない」状態なので除外する。

    Args:
        ax:     matplotlib Axes オブジェクト。
        df_old: 旧型機の fleet データ DataFrame。
    """
    df_plot = df_old[df_old['Status'] != STATUS_ACTIVE]
    pivot   = df_plot.groupby(['Month_Bin', 'Status'], observed=False).size().unstack(fill_value=0)
    for col in [STATUS_CENSORED, STATUS_FAILED]:
        if col not in pivot:
            pivot[col] = 0
    pivot    = pivot.reindex(range(1, EVAL_MONTHS_OLD + 1), fill_value=0)
    cum_data = pivot.cumsum()

    ax.bar(cum_data.index, cum_data[STATUS_CENSORED],
           color='skyblue', label='予防保守 (Censored)')
    ax.bar(cum_data.index, cum_data[STATUS_FAILED],
           bottom=cum_data[STATUS_CENSORED],
           color='red', alpha=0.5, label='事後保守 (Failed)')

    ax.set_title('② 旧型の現場実績 (累積部品交換数)')
    ax.set_xlim(0, EVAL_MONTHS_OLD + 1)
    ax.set_xlabel('フリート稼働開始からの経過月数')
    ax.set_ylabel('累積交換数 (個)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')  # Failed を上に表示


def plot_ax2_new_fleet_history(ax, df_new: pd.DataFrame) -> None:
    """② 新型機の早期評価 (累積部品交換数) を積み上げ棒グラフで描画する。

    ① と同じ形式で新型機の早期稼働データを可視化する。
    各棒の上に月次件数ラベルを表示する。

    Args:
        ax:     matplotlib Axes オブジェクト。
        df_new: 新型機の fleet データ DataFrame。
    """
    df_plot = df_new[df_new['Status'] != STATUS_ACTIVE].copy()
    pivot   = df_plot.groupby(['Month_Bin', 'Status'], observed=False).size().unstack(fill_value=0)
    for col in [STATUS_CENSORED, STATUS_FAILED]:
        if col not in pivot:
            pivot[col] = 0
    pivot    = pivot.reindex(range(1, EVAL_MONTHS_NEW + 1), fill_value=0)
    cum_data = pivot.cumsum()

    fontsize = 6
    p_cens = ax.bar(cum_data.index, cum_data[STATUS_CENSORED],
                    color='skyblue', label='予防保守 (Censored)')
    p_fail = ax.bar(cum_data.index, cum_data[STATUS_FAILED],
                    bottom=cum_data[STATUS_CENSORED],
                    color='red', alpha=0.5, label='事後保守 (Failed)')
    ax.bar_label(p_cens, label_type='center', fontsize=fontsize, color='black')
    ax.bar_label(p_fail, label_type='center', fontsize=fontsize, color='black')

    ax.set_title('③ 新型の早期評価 (累積部品交換数)')
    ax.set_xlim(0, EVAL_MONTHS_NEW + 1)
    ax.set_xlabel('フリート稼働開始からの経過月数')
    ax.set_ylabel('累積交換数 (個)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')


def plot_ax3_bathtub_curve(ax) -> None:
    """③ 旧型のバスタブ曲線の模式図 (混合ワイブルの解析的ハザード関数) を描画する。

    ■データ点に基づくMLE推定を不採用とした理由
        市場 Failed データから 3成分を最尤推定法 (MLE) で逆推定すると
        局所解への収束が不安定で、教科書的な形状が再現されなかった。
        WEIBULL_MODES に真のパラメータが既知であるため、
        解析的に計算する方が正確かつ安定している。

    ■スケーリング係数の復元
        generate_fleet_data() と同じシード (42) で raw_lives を再生成し、
        データ生成時と同一の scaling_factor = B10_DESIGN_OLD / raw_b10 を得る。

    ■模式化のためのスケーリング
        各成分を「ピーク=1」に正規化後、混合比 prob を掛けることで
        「ピーク高さが混合比に比例した模式図」を描く。
        合成バスタブはスケール後の成分の和として表現する。

    Args:
        ax: matplotlib Axes オブジェクト。
    """
    # スケーリング係数の復元 (generate_fleet_data と同一シード)
    np.random.seed(42)
    POOL_SIZE    = 15000
    mode_indices = np.random.choice(
        len(WEIBULL_MODES), size=POOL_SIZE,
        p=[m['prob'] for m in WEIBULL_MODES]
    )
    raw_lives = np.zeros(POOL_SIZE)
    for i, mode in enumerate(WEIBULL_MODES):
        mask   = (mode_indices == i)
        n_mode = int(np.sum(mask))
        if n_mode > 0:
            raw_lives[mask] = np.random.weibull(mode['beta'], n_mode)

    raw_b10        = np.percentile(raw_lives, 10)
    scaling_factor = B10_DESIGN_OLD / raw_b10

    betas = [m['beta'] for m in WEIBULL_MODES]
    probs = [m['prob']  for m in WEIBULL_MODES]
    etas  = [scaling_factor] * len(WEIBULL_MODES)  # 全成分共通 η

    # 描画範囲 (t=0 は β<1 で発散するため除外)
    x_min  = scaling_factor * 0.0001
    x_max  = PM_CYCLES_OLD * 1.5
    x_vals = np.linspace(x_min, x_max, 2000)

    def _weibull_hazard(t, beta, eta):
        """単一ワイブル成分のハザード関数: h(t) = (β/η) * (t/η)^(β-1)"""
        return (beta / eta) * (t / eta) ** (beta - 1)

    # 各成分: ピーク正規化 → 混合比でスケール
    h_scaled = []
    for beta, eta, prob in zip(betas, etas, probs):
        h_raw = _weibull_hazard(x_vals, beta, eta)
        peak  = np.max(h_raw)
        h_scaled.append((h_raw / peak) * prob if peak > 0 else np.zeros_like(h_raw))

    h_bathtub = sum(h_scaled)  # 合成バスタブ = 模式スケール後の成分の和

    mode_styles = [
        {'label': '初期故障 (β<1)', 'color': 'royalblue', 'ls': '--'},
        {'label': '偶発故障 (β=1)', 'color': 'green',     'ls': '-.'},
        {'label': '摩耗故障 (β>1)', 'color': 'tomato',    'ls': ':'},
    ]
    for h_s, style, prob in zip(h_scaled, mode_styles, probs):
        ax.plot(x_vals, h_s, color=style['color'], linestyle=style['ls'],
                linewidth=2.0, label=f"{style['label']} (混合比={prob:.0%})", zorder=3)

    ax.plot(x_vals, h_bathtub, color='black', linewidth=2.5,
            label='バスタブ (合成)', zorder=4)
    ax.axvline(B10_DESIGN_OLD, color=COLOR_OLD, linestyle='--', linewidth=1.2,
               alpha=0.8, label=f'B10設計目標 {B10_DESIGN_OLD // 1000}k')

    ax.legend(loc='upper right', fontsize='x-small')
    ax.set_title('① 旧型のバスタブ曲線の模式図 (混合ワイブル)')
    ax.set_xlabel('稼働時間 [サイクル]')
    ax.set_ylabel('ハザード率 h(t)  ※模式値')
    ax.set_ylim(bottom=0)
    ax.set_yticks([])  # 模式図のため数値目盛りは非表示
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kcycles))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100000))
    ax.grid(True, which='major', ls='-', alpha=0.2)


def plot_ax4_failure_life_distribution(ax, df_old: pd.DataFrame, df_new: pd.DataFrame) -> None:
    """④ 部品ライフ実力値の比較 - 事後保守 (故障) データのみカーネル密度推定 (KDE) で可視化する。

    Failed データのみを使うことで、予防保守により隠れた実際の故障寿命分布を比較する。
    B10 は 10 パーセンタイルで算出する (Censored を含まない粗い推定)。

    Args:
        ax:     matplotlib Axes オブジェクト。
        df_old: 旧型機の fleet データ。
        df_new: 新型機の fleet データ (全期間)。
    """
    x_grid   = np.linspace(0, B10_DESIGN_NEW * CENSORING_FACTOR * 2, 500)
    fail_old = df_old[df_old['Status'] == STATUS_FAILED]['Observed_Cycles']
    fail_new = df_new[df_new['Status'] == STATUS_FAILED]['Observed_Cycles']

    if len(fail_old) > 1:
        kde_old = stats.gaussian_kde(fail_old)(x_grid)
        ax.plot(x_grid, kde_old, color=COLOR_OLD, label='旧型(事後保守)')
        ax.fill_between(x_grid, kde_old, alpha=0.2, color=COLOR_OLD)

    if len(fail_new) > 1:
        kde_new = stats.gaussian_kde(fail_new)(x_grid)
        ax.plot(x_grid, kde_new, color=COLOR_NEW, label='新型(事後保守)')
        ax.fill_between(x_grid, kde_new, alpha=0.2, color=COLOR_NEW)

    b10_old = float(np.percentile(fail_old, 10))
    b10_new = float(np.percentile(fail_new, 10))
    ax.axvline(b10_old, color=COLOR_OLD, linestyle='--', alpha=1.0)
    ax.axvline(b10_new, color=COLOR_NEW, linestyle='--', alpha=1.0)
    ax.text(b10_old, ax.get_ylim()[1] * 0.20,
            f'旧型 10% {int(b10_old / 1000)}k', color='darkblue', ha='left', fontsize=9)
    ax.text(b10_new, ax.get_ylim()[1] * 0.10,
            f'新型 10% {int(b10_new / 1000)}k', color='darkblue', ha='left', fontsize=9)

    ax.set_title('④ 部品ライフ実力値の比較 - 事後保守(故障)のみ')
    ax.set_xlabel('稼働時間 [サイクル]')
    ax.set_xlim(0, B10_DESIGN_NEW * CENSORING_FACTOR * 1.2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kcycles))
    ax.legend()


def plot_ax5_pm_distribution(ax, df_old: pd.DataFrame, df_new: pd.DataFrame) -> None:
    """⑤ 保守運用の比較 - 予防保守 (打ち切り) データのみカーネル密度推定 (KDE) で可視化する。

    Censored データの分布が予防保守目標サイクル付近に集中しているかを確認する。
    分布が目標から大きくずれている場合、現場の運用逸脱を示す可能性がある。

    Args:
        ax:     matplotlib Axes オブジェクト。
        df_old: 旧型機の fleet データ。
        df_new: 新型機の fleet データ。
    """
    x_grid   = np.linspace(0, B10_DESIGN_NEW * CENSORING_FACTOR * 2, 500)
    cens_old = df_old[df_old['Status'] == STATUS_CENSORED]['Observed_Cycles']
    cens_new = df_new[df_new['Status'] == STATUS_CENSORED]['Observed_Cycles']

    if len(cens_old) > 1:
        kde_old = stats.gaussian_kde(cens_old)(x_grid)
        ax.plot(x_grid, kde_old, color=COLOR_OLD, label='旧型(予防保守)')
        ax.fill_between(x_grid, kde_old, alpha=0.2, color=COLOR_OLD)

    if len(cens_new) > 1:
        kde_new = stats.gaussian_kde(cens_new)(x_grid)
        ax.plot(x_grid, kde_new, color='blue', label='新型(予防保守)')
        ax.fill_between(x_grid, kde_new, alpha=0.2, color='blue')

    ax.axvline(PM_CYCLES_OLD, color=COLOR_OLD, linestyle=':', linewidth=2)
    ax.axvline(PM_CYCLES_NEW, color=COLOR_NEW, linestyle=':', linewidth=2)
    ax.text(PM_CYCLES_OLD, ax.get_ylim()[1] * 0.9,
            f'旧型 交換目標 {PM_CYCLES_OLD / 1000:.0f}k', color='darkblue', ha='right')
    ax.text(PM_CYCLES_NEW, ax.get_ylim()[1] * 0.9,
            f'新型 交換目標 {PM_CYCLES_NEW / 1000:.0f}k', color='darkblue', ha='left')

    ax.set_title('⑤ 保守運用の比較 - 予防保守(打ち切り)のみ')
    ax.set_xlabel('稼働時間 [サイクル]')
    ax.set_xlim(0, B10_DESIGN_NEW * CENSORING_FACTOR * 1.2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kcycles))
    ax.legend(loc='center left')


def plot_ax6_km_survival_curves(ax, df_old: pd.DataFrame, df_new: pd.DataFrame):
    """⑥ 市場生存率の実績比較 (Kaplan-Meier 曲線) を描画する。

    Censored データを適切に考慮した KM 推定により、
    予防保守の多い環境でも真の生存率を推定する。
    B10 (生存率 90% に対応する寿命) の交点をマークする。

    Args:
        ax:     matplotlib Axes オブジェクト。
        df_old: 旧型機の fleet データ。
        df_new: 新型機の fleet データ。

    Returns:
        (t_old, s_old, t_new, s_new): KM 推定の時刻・生存確率配列 (⑦以降で再利用)。
    """
    events_old = (df_old['Status'] == STATUS_FAILED).values
    t_old, s_old, l_old, u_old = compute_km_with_ci(
        df_old['Observed_Cycles'].values, events_old
    )
    ax.step(t_old, s_old, where='post', color=COLOR_OLD, alpha=1.0, label='旧型 (実績)')
    ax.fill_between(t_old, l_old, u_old, step='post', color=COLOR_OLD, alpha=0.4)

    events_new = (df_new['Status'] == STATUS_FAILED).values
    t_new, s_new, l_new, u_new = compute_km_with_ci(
        df_new['Observed_Cycles'].values, events_new
    )
    ax.step(t_new, s_new, where='post', color=COLOR_NEW, linewidth=1.5, label='新型 (実績)')
    ax.fill_between(t_new, l_new, u_new, step='post', color=COLOR_NEW, alpha=0.1)

    ax.axhline(0.9, color='red', linestyle=':', alpha=0.6)  # B10 基準線 (生存率 90%)

    def _mark_b10_on_km(times, survival, color, label):
        """KM 曲線と B10 基準線 (S=0.9) の交点をマークする。"""
        if min(survival) <= 0.9:
            b10_est = float(np.interp(0.9, survival[::-1], times[::-1]))
            ax.vlines(b10_est, 0, 0.9, color=color, linestyle='--', alpha=0.7)
            ax.plot(b10_est, 0.9, 'o', color=color)
            ha = 'right' if label == '旧型' else 'left'
            ax.text(b10_est, 0.5,
                    f'{label}\n90% {int(b10_est / 1000)}k',
                    color=color, ha=ha, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    _mark_b10_on_km(t_old, s_old, COLOR_OLD, '旧型')
    _mark_b10_on_km(t_new, s_new, COLOR_NEW, '新型')

    ax.set_title('⑥ 市場生存率の実績比較 (KM曲線)')
    ax.set_xlabel('稼働時間 [サイクル]')
    ax.set_ylabel('生存率')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, B10_DESIGN_NEW * CENSORING_FACTOR * 1.2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kcycles))
    ax.grid(alpha=0.3)
    ax.legend()

    return np.array(t_old), np.array(s_old), np.array(t_new), np.array(s_new)


def plot_ax7_equivalence_test(ax,
                              b10_samples_new: np.ndarray,
                              b10_samples_old: np.ndarray,
                              b10_old_baseline: float) -> None:
    """⑦ 同等性検証: B10 ライフ差の信頼区間 vs 許容限界 を描画する。

    ■同等性検証ロジック (TOST: Two One-Sided Tests)
        差 D = B10(新型) - B10(旧型) のブートストラップ分布を生成し、
        90% 信頼区間 [CI_lower, CI_upper] を算出する。(CI: confidence interval)

        判定: CI_lower ≧ 許容下限 (EQUIV_MARGIN_LOWER_RATIO × 旧型B10) であれば
              「同等以上」と判定する。上限は参考表示のみ。

    ■旧型の不確実性を考慮する意図:
        旧型を点推定で扱うと旧型自体の推定誤差が無視される。
        旧型にもブートストラップを適用してペアワイズ差の分布を構築することで、
        両者の不確実性を同時に反映した同等性検証が可能となる。

    Args:
        ax:               matplotlib Axes オブジェクト。
        b10_samples_new:  新型 B10 のブートストラップサンプル配列。
        b10_samples_old:  旧型 B10 のブートストラップサンプル配列。
        b10_old_baseline: 旧型 B10 の中央値 (許容限界計算の基準値)。
    """
    sufficient_new = len(b10_samples_new) > 10
    sufficient_old = len(b10_samples_old) > 10

    if sufficient_new and sufficient_old:
        # ペアワイズ差の分布を生成
        n_pair  = min(len(b10_samples_new), len(b10_samples_old))
        idx_new = np.random.choice(len(b10_samples_new), n_pair, replace=False)
        idx_old = np.random.choice(len(b10_samples_old), n_pair, replace=False)
        diff_samples = b10_samples_new[idx_new] - b10_samples_old[idx_old]

        ci_lower_diff, ci_upper_diff = np.percentile(diff_samples, [5, 95])
        equiv_margin_lower = b10_old_baseline * EQUIV_MARGIN_LOWER_RATIO
        equiv_margin_upper = b10_old_baseline * EQUIV_MARGIN_UPPER_RATIO
        is_equivalent      = ci_lower_diff >= equiv_margin_lower  # 主判定

        # カーネル密度推定 (KDE) 描画
        kde_diff   = stats.gaussian_kde(diff_samples)
        x_plot_min = min(diff_samples.min(), equiv_margin_lower) * 1.5
        x_plot_max = max(diff_samples.max(), equiv_margin_upper) * 1.5
        x_vals     = np.linspace(x_plot_min, x_plot_max, 600)
        y_vals     = kde_diff(x_vals)
        y_peak     = float(np.max(y_vals))

        ax.fill_between(x_vals, 0, y_vals, color='steelblue', alpha=0.25,
                        label='差の分布 (新型 - 旧型)')
        ax.plot(x_vals, y_vals, color='steelblue', linewidth=1.5)

        # 90% CI 強調
        mask_ci    = (x_vals >= ci_lower_diff) & (x_vals <= ci_upper_diff)
        ci_x_lower = float(np.min(x_vals[mask_ci]))
        ci_x_upper = float(np.max(x_vals[mask_ci]))
        ci_y_peak  = float(np.max(y_vals[mask_ci]))
        ax.fill_between(x_vals[mask_ci], 0, y_vals[mask_ci],
                        color='steelblue', alpha=0.4, label='90% 信頼区間')

        # CI 境界線とラベル
        for ci_val, ha, side in [(ci_lower_diff, 'right', '下限'),
                                  (ci_upper_diff, 'left',  '上限')]:
            ax.axvline(ci_val, ymin=0, ymax=0.6, color='steelblue', linestyle='-', linewidth=1)
            ax.text(ci_val, ci_y_peak * 0.55,
                    f'90%信頼限界 {side}\n{int(ci_val / 1000)}k',
                    color='steelblue', ha=ha, fontsize='small', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.0', facecolor='white',
                              alpha=0.7, edgecolor='none'))

        # CI バー
        ax.add_patch(patches.Rectangle(
            (ci_x_lower, 0), ci_x_upper - ci_x_lower, y_peak * 0.05,
            linewidth=0, facecolor='steelblue', alpha=1.0
        ))

        # 許容範囲の緑帯
        ax.add_patch(patches.Rectangle(
            (equiv_margin_lower, 0),
            equiv_margin_upper - equiv_margin_lower, y_peak * 1.2,
            linewidth=0, facecolor='green', alpha=0.3
        ))
        # 許容限界ラベル
        label_y = y_peak * 0.7
        ax.text(equiv_margin_lower, label_y,
                f'許容限界 下限 ({EQUIV_MARGIN_LOWER_RATIO * 100:.0f}%)\n'
                f'{int(equiv_margin_lower / 1000)}k',
                color='green', ha='right', fontsize='small', weight='bold',
                bbox=dict(boxstyle='round,pad=0.0', facecolor='white',
                          alpha=0.7, edgecolor='none'))
        ax.text(equiv_margin_upper, label_y,
                f'許容限界 上限 (+{EQUIV_MARGIN_UPPER_RATIO * 100:.0f}%)\n'
                f'{int(equiv_margin_upper / 1000)}k',
                color='green', ha='left', fontsize='small', weight='bold',
                bbox=dict(boxstyle='round,pad=0.0', facecolor='white',
                          alpha=0.7, edgecolor='none'))

        # NG 時のみ: CI 下限と許容下限を結ぶ矢印
        if not is_equivalent:
            y_arrow = y_peak * 0.20
            ax.annotate('', xy=(equiv_margin_lower, y_arrow), xytext=(ci_lower_diff, y_arrow),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2.0))

        # 判定結果テキスト
        prob_better   = np.mean(b10_samples_new[idx_new] >= b10_samples_old[idx_old]) * 100
        verdict_color = 'green' if is_equivalent else 'red'
        verdict_head  = 'OK 同等以上' if is_equivalent else 'NG 同等以上と言い切れない'
        verdict_detail = (
            f'CI下限 ({int(ci_lower_diff / 1000)}k) ≧ 許容下限 ({int(equiv_margin_lower / 1000)}k)'
            if is_equivalent else
            f'CI下限 ({int(ci_lower_diff / 1000)}k) < 許容下限 ({int(equiv_margin_lower / 1000)}k)'
        )
        msg = (f'{verdict_head}\n'
               f'{verdict_detail}\n'
               f'旧型以上の確率: {prob_better:.1f}%\n'
               f'(旧型B10基準値: {int(b10_old_baseline / 1000)}k, '
               f'許容マージン{EQUIV_MARGIN_LOWER_RATIO * 100:.0f}%)')
        ax.text(0.5, 0.4, msg, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95,
                          edgecolor=verdict_color, linewidth=2))

    elif sufficient_new:
        # 旧型サンプル不足時: 新型分布のみ表示
        b10_median = float(np.median(b10_samples_new))
        kde_new    = stats.gaussian_kde(b10_samples_new)
        x_vals     = np.linspace(b10_samples_new.min() * 0.7, b10_samples_new.max() * 1.3, 500)
        ax.plot(x_vals, kde_new(x_vals), color=COLOR_NEW, linewidth=2)
        ax.fill_between(x_vals, 0, kde_new(x_vals), color=COLOR_NEW, alpha=0.2)
        ax.axvline(b10_median, color=COLOR_NEW, linestyle='--',
                   label=f'新型 B10推計: {int(b10_median / 1000)}k')
        ax.axvline(b10_old_baseline, color=COLOR_OLD, linestyle='--', linewidth=2,
                   label=f'旧型 B10実績: {int(b10_old_baseline / 1000)}k')
        ax.text(0.02, 0.97, '旧型ブートストラップサンプル不足\n新型分布のみ表示',
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    else:
        ax.text(0.5, 0.5, 'データ不足\n(Failed < 3件)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray')

    ax.set_xlim(-(PM_CYCLES_NEW - PM_CYCLES_OLD), (PM_CYCLES_NEW - PM_CYCLES_OLD) * 2)
    ax.set_yticks([])
    ax.set_xlabel('B10ライフ差 [サイクル]  (新型 - 旧型)')
    ax.set_title('⑦ 同等性検証: B10ライフ差の信頼区間 vs 許容限界')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kcycles))


def plot_ax8_saturation_chart(ax, df_new: pd.DataFrame, b10_old_baseline: float) -> None:
    """⑧ 意思決定サチレーション (B10 予測の月次収束) を描画する。

    評価月数が増えるにつれて B10 推定値の信頼区間がどう収束するかを可視化する。
    「いつ意思決定できるか」を判断するためのチャートである。

    ■フォールバック:
        初期は Failed 件数が少なく局所フィットが機能しない。
        そのため局所フィットが不十分な月は全域フィット (fit_weibull_simple) で補完し、
        「少 N 期の参考値」として区別できるように色を変えて表示する。

    Args:
        ax:               matplotlib Axes オブジェクト。
        df_new:           新型機の fleet データ (全期間)。
        b10_old_baseline: 旧型 B10 中央値 (水平参照線として描画)。
    """
    months = np.arange(2, EVAL_MONTHS_NEW + 1)
    medians, lowers, uppers, is_local_fit = [], [], [], []

    for m in months:
        df_month = df_new[df_new['Event_Month'] <= m]
        samples  = bootstrap_b10_local(df_month, n_boot=N_BOOTSTRAP_MAIN,
                                       f_upper_limit=LOCAL_FIT_F_LIMIT)

        if len(samples) > 15:
            # 局所フィット成功
            medians.append(float(np.median(samples)))
            lo, hi = np.percentile(samples, [5, 95])
            lowers.append(lo)
            uppers.append(hi)
            is_local_fit.append(True)
        else:
            # 全域フィットで補完
            beta_f, eta_f = fit_weibull_simple(
                df_month['Observed_Cycles'].values,
                (df_month['Status'] == STATUS_FAILED).values
            )
            if beta_f is not None:
                b10_f = eta_f * (-np.log(0.9)) ** (1 / beta_f)
                medians.append(b10_f)
            else:
                medians.append(float('nan'))
            lowers.append(float('nan'))
            uppers.append(float('nan'))
            is_local_fit.append(False)

    medians      = np.array(medians)
    is_local_fit = np.array(is_local_fit)

    # 全域フィット期間 (参考表示)
    global_mask   = ~is_local_fit
    valid_globals = [x for x in medians[global_mask] if not math.isnan(x)]
    if len(valid_globals) >= 1:
        ax.plot(months[global_mask], medians[global_mask],
                'o--', color='gray', alpha=0.5,
                label='新型 (全体フィット; 少N期の参考)')

    # 局所フィット期間 (主表示)
    ax.plot(months[is_local_fit], medians[is_local_fit],
            'o-', color=COLOR_NEW, linewidth=2,
            label='新型 (初期故障領域フィット;\n       バンドは90%信頼区間)')
    ax.fill_between(months, lowers, uppers, color=COLOR_NEW, alpha=0.15)

    ax.axhline(B10_DESIGN_NEW, color=COLOR_NEW, alpha=0.3, linestyle='--',
               label=f'新型 B10目標ライフ {B10_DESIGN_NEW // 1000}k')
    ax.axhline(b10_old_baseline, color=COLOR_OLD, linestyle='--',
               label=f'旧型 B10実績ライフ {int(b10_old_baseline / 1000)}k')

    valid_uppers = [y for y in uppers if not math.isnan(y)]
    ax.set_ylim(0, max(valid_uppers) if valid_uppers else B10_DESIGN_NEW * 2)
    ax.set_xlim(0, EVAL_MONTHS_NEW + 1)
    ax.set_title('⑧ 意思決定サチレーション (B10予測の収束)')
    ax.set_xlabel('稼働経過月数')
    ax.set_ylabel('B10予測値 [サイクル]')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper left', fontsize='small')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kcycles))


def plot_ax9_broken_weibull(ax, df_old: pd.DataFrame, df_new_current: pd.DataFrame) -> None:
    """⑨ 折れ線ワイブルプロット (旧型・新型の初期故障領域フィット比較) を描画する。

    ワイブル確率紙上に KM 推定点をプロットし、累積故障率 30% 以下の
    初期故障領域に線形回帰を行う。回帰直線の傾きが β 推定値となる。
    信頼区間バンドは Student-t 分布を用いた線形回帰の予測区間として算出する。

    Args:
        ax:             matplotlib Axes オブジェクト。
        df_old:         旧型機の fleet データ (全期間)。
        df_new_current: 新型機の fleet データ (評価月時点まで)。
    """
    EARLY_FAILURE_F_LIMIT = 0.30  # 初期故障領域の境界 (累積故障率 30%)

    def _plot_single_weibull(df_target: pd.DataFrame, color: str, label: str):
        """単一機種のワイブルプロットと回帰線を描画する。"""
        durations       = df_target['Observed_Cycles'].values
        events          = (df_target['Status'] == STATUS_FAILED).values
        times, survival = compute_km_simple(durations, events)
        cum_failure     = 1 - survival

        valid = (cum_failure > 0) & (survival > 0)
        if not np.any(valid):
            return

        t_valid = times[valid]
        f_valid = cum_failure[valid]
        X_all   = np.log(t_valid)
        Y_all   = np.log(-np.log(1 - f_valid))

        ax.scatter(X_all, Y_all, color=color, alpha=1, s=5)

        # 初期故障領域 (F ≤ 30%) の回帰 + 95% 予測区間バンド
        mask_early = (f_valid <= EARLY_FAILURE_F_LIMIT)
        n_early    = int(np.sum(mask_early))

        if n_early >= 3:
            X_e, Y_e = X_all[mask_early], Y_all[mask_early]
            result   = stats.linregress(X_e, Y_e)
            slope, intercept = result.slope, result.intercept

            ax.plot(X_e, slope * X_e + intercept, color=color, linewidth=1,
                    label=f'{label} (初期故障領域 β={slope:.1f})')

            # Student-t による 95% 予測区間
            t_crit = stats.t.ppf(0.975, n_early - 2)
            Sxx    = float(np.sum((X_e - X_e.mean()) ** 2))
            if Sxx > 0:
                s_res  = float(np.sqrt(np.sum((Y_e - (slope * X_e + intercept)) ** 2) / (n_early - 2)))
                x_band = np.linspace(X_e.min() - 0.1, X_e.max() + 0.1, 50)
                y_hat  = slope * x_band + intercept
                ci_hw  = t_crit * s_res * np.sqrt(
                    1 / n_early + (x_band - X_e.mean()) ** 2 / Sxx
                )
                ax.fill_between(x_band, y_hat - ci_hw, y_hat + ci_hw,
                                color=color, alpha=0.15)

            # B10 交点マーク (β > 0 の場合のみ)
            if slope > 0:
                b10_y = np.log(-np.log(0.9))
                b10_x = (b10_y - intercept) / slope
                ax.axhline(y=b10_y, color='red', linestyle=':', alpha=0.5)
                ax.axvline(x=b10_x, color='red', linestyle=':', alpha=0.5)
                label_y = weibull_plot_y(80) if label == '旧型' else weibull_plot_y(50)
                ha      = 'right'             if label == '旧型' else 'left'
                ax.text(b10_x, label_y,
                        f'{label} B10={np.exp(b10_x) / 1000:.1f}k',
                        color='red', ha=ha, fontsize=9, fontweight='bold')

        elif n_early == 2:
            # 2 点のみ: CI 計算不能のため直線のみ
            slope, intercept = np.polyfit(X_all[mask_early], Y_all[mask_early], 1)
            ax.plot(X_all[mask_early], slope * X_all[mask_early] + intercept,
                    color=color, linewidth=3,
                    label=f'{label} (初期故障領域 β={slope:.1f})')

        # 偶発・摩耗領域 (F > 30%) の補助線
        mask_late = (f_valid > EARLY_FAILURE_F_LIMIT)
        if int(np.sum(mask_late)) >= 2:
            slope2, intercept2 = np.polyfit(X_all[mask_late], Y_all[mask_late], 1)
            ax.plot(X_all[mask_late], slope2 * X_all[mask_late] + intercept2,
                    color=color, linestyle='--', linewidth=1.0, alpha=0.7)

    _plot_single_weibull(df_old,         COLOR_OLD, '旧型')
    _plot_single_weibull(df_new_current, COLOR_NEW, '新型')

    y_ticks_pct = np.array([0.1, 1, 5, 10, 30, 50, 80, 95, 99])
    ax.set_yticks(weibull_plot_y(y_ticks_pct))
    ax.set_yticklabels([f'{v}%' for v in y_ticks_pct])
    ax.set_ylim(weibull_plot_y(0.1), weibull_plot_y(99.9))
    ax.set_title('⑨ 折れ線ワイブル')
    ax.set_ylabel('累積故障率')
    ax.set_xlabel('稼働時間 [サイクル]')
    ax.legend(loc='upper left', fontsize='small')


# ==============================================================================
# メイン処理
# ==============================================================================

def main() -> None:
    """ダッシュボード全体を生成するエントリーポイント。

    ■ストーリーと対応するグラフ構成 (3行 × 3列 GridSpec)

        このダッシュボードは、現場でワイブル分析に馴染んだ者が
        「新型機の部品ライフが旧型機と同等以上か」を早期に判断するための
        一連のストーリーとして読めるよう、グラフ配置を設計している。

        上段 - 前提モデルの提示と市場実績の確認
            ① バスタブ曲線 (gs[0,0]) : ストーリーの起点。混合ワイブルモデルを提示し、
                                        シミュレーション全体の前提を共有する。
            ② 旧型実績    (gs[0,1]) : バスタブモデルから生成した旧型フリートの稼働実績。
            ③ 新型実績    (gs[0,2]) : 評価対象である新型フリートの初期稼働実績。

        中段 - 現場データの詳細把握
            ④ 故障ライフ分布 (gs[1,0]) : Failed データから部品寿命の実力値を比較。
            ⑤ PM 分布       (gs[1,1]) : Censored データから予防保守の運用実態を把握。
            ⑥ KM 生存曲線   (gs[1,2]) : KM 法で Censored を考慮した生存率。
                                         Weibull 回帰へつなげる。

        下段 - Weibull 回帰によるB10推定・判断・確認
            ⑦ 同等性検証      (gs[2,0]) : ブートストラップ B10 差の CI vs 許容限界 (TOST)。
            ⑧ サチレーション  (gs[2,1]) : 何ヶ月データを集めれば同等性を判断できるかを可視化。
            ⑨ ワイブルプロット (gs[2,2]) : Weibull 確率紙で推定結果を目視確認。

    ■処理フロー
        1. 旧型・新型のフリートデータを生成
        2. 上段グラフを描画
        3. 中段グラフを描画
           - 旧型 B10 基準値をブートストラップで算出 (⑦⑧ で共用)
           - 新型 B10 ブートストラップサンプルを算出 (⑦ で使用)
        4. 下段グラフを描画
    """
    # --- 1. データ生成 ---
    df_old         = generate_fleet_data(is_new=False)
    df_new         = generate_fleet_data(is_new=True)
    df_new_current = df_new[df_new['Event_Month'] <= EVAL_MONTHS_NEW]

    print('=== 新型機 データサマリ (評価期間内) ===')
    print(df_new_current.groupby('Status')['Observed_Cycles'].describe())

    # --- 2. 図・レイアウト初期化 ---
    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.4, wspace=0.3)
    fig.suptitle(
        f'早期同等性評価ダッシュボード  '
        f'(新型 {EVAL_MONTHS_NEW}ヶ月経過  '
        f'B10設計目標={B10_DESIGN_NEW / 1000:.0f}k  '
        f'予防保守目標={PM_CYCLES_NEW / 1000:.0f}k)',
        fontsize=18, fontweight='bold'
    )

    # --- 3. 上段グラフ: 前提モデルの提示 → 旧型実績 → 新型実績 ---
    # ① バスタブ曲線を左端に配置: ストーリーの起点として最初に目に入る位置
    plot_ax3_bathtub_curve(fig.add_subplot(gs[0, 0]))
    # ② 旧型実績: バスタブモデルから生成された市場データの全体像
    plot_ax1_old_fleet_history(fig.add_subplot(gs[0, 1]), df_old)
    # ③ 新型実績: 評価対象の初期稼働データ
    plot_ax2_new_fleet_history(fig.add_subplot(gs[0, 2]), df_new)

    # --- 4. 中段グラフ: Failed / Censored の詳細 → KM で統合 ---
    plot_ax4_failure_life_distribution(fig.add_subplot(gs[1, 0]), df_old, df_new)
    plot_ax5_pm_distribution(fig.add_subplot(gs[1, 1]), df_old, df_new)
    plot_ax6_km_survival_curves(fig.add_subplot(gs[1, 2]), df_old, df_new)

    # --- 5. 旧型 B10 基準値の算出 (⑦⑧ 共用) ---
    b10_samples_old  = bootstrap_b10_local(df_old, n_boot=N_BOOTSTRAP_OLD,
                                           f_upper_limit=LOCAL_FIT_F_LIMIT)
    b10_old_baseline = (float(np.median(b10_samples_old))
                        if len(b10_samples_old) > 0 else float(B10_DESIGN_OLD))
    print(f'\n旧型 B10 基準値 (中央値): {b10_old_baseline:.0f} サイクル '
          f'({len(b10_samples_old)} サンプル)')

    # --- 6. 新型 B10 ブートストラップサンプルの算出 (⑦ 使用) ---
    b10_samples_new = bootstrap_b10_local(df_new_current, n_boot=N_BOOTSTRAP_MAIN,
                                          f_upper_limit=LOCAL_FIT_F_LIMIT)
    print(f'新型 B10 サンプル数: {len(b10_samples_new)}')

    # --- 7. 下段グラフ: Weibull 回帰による推定 → 同等性判断 → ワイブルプロットで確認 ---
    plot_ax7_equivalence_test(
        fig.add_subplot(gs[2, 0]),
        b10_samples_new, b10_samples_old, b10_old_baseline
    )
    plot_ax8_saturation_chart(
        fig.add_subplot(gs[2, 1]),
        df_new, b10_old_baseline
    )
    plot_ax9_broken_weibull(
        fig.add_subplot(gs[2, 2]),
        df_old, df_new_current
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


# ==============================================================================
# エントリーポイント
# ==============================================================================

if __name__ == '__main__':
    # multiprocessing との互換性確保 (PyMC 等のライブラリ利用時の保険的対応)
    if '__spec__' not in globals():
        __spec__ = None

    main()
