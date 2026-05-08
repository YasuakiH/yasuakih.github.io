# fleet_reliability_simulator.py

'''
Fleet Early Equivalence Testing Dashboard:
This simulation dashboard was developed to enable quality control personnel practicing quality engineering and reliability engineering to quickly assess whether the component lifespan of new machine models is equivalent to or exceeds that of older models during the initial deployment phase, when operational data is extremely limited.

フリート早期同等性検証ダッシュボード:
品質工学や信頼性工学を実践する品質管理担当者が「稼働データが極めて少ない導入初期段階において、新型機の部品寿命が旧型機と同等以上か」を早期に評価するために構築されたシミュレーション・ダッシュボードである。
'''

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import warnings
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from reliability.Distributions import Weibull_Distribution

# 日本語フォント設定 (環境に合わせる)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Arial']

# シミュレーション用パラメータ
EVAL_MONTHS_NEW = 12
# EVAL_MONTHS_NEW = 24
# EVAL_MONTHS_NEW = 36
# EVAL_MONTHS_NEW = 48
# EVAL_MONTHS_NEW = 60
# EVAL_MONTHS_NEW = 72
EVAL_MONTHS_OLD = 72

# 予防保守目標 (B10設計ライフの何倍で打ち切るか)
CENSORING_FACTOR = 4.0
# CENSORING_FACTOR = 8.0

CYCLES_PER_DAY = 4000    # 日あたりの部品稼働サイクル数

B10_TARGET_NEW = 100000  # 設計ライフ(新型)
B10_TARGET_OLD = 80000   # 設計ライフ(旧型)
PM_TARGET_NEW = B10_TARGET_NEW * CENSORING_FACTOR  # 予防保守目標(新型) = 設計ライフ(新型) x CENSORING_FACTOR
PM_TARGET_OLD = B10_TARGET_OLD * CENSORING_FACTOR  # 予防保守目標(旧型) = 設計ライフ(旧型) x CENSORING_FACTOR

# 競合モデルのワイブルパラメータ (シミュレーション用データ作成時、および③バスタブ曲線で参照する)
#   beta : β値
#   prob : 生成確率
# * alphaは持たない。理由: B10からスケーリングするため不要。
WEIBULL_MODES = [
    {'beta': 0.7, 'prob': 0.3},  # 初期故障モード β<1
    {'beta': 1.0, 'prob': 0.4},  # 偶発故障モード β=1
    {'beta': 2.5, 'prob': 0.3},  # 摩耗故障モード β>1
]
print(f'WEIBULL_MODES={WEIBULL_MODES}')


# 軸ラベルを k 単位にフォーマット
def cycle_formatter(x, pos):
    return f'{int(x/1000)}k'


# ==========================================
# データシミュレーション・エンジン (混合ワイブルモデル)
# ==========================================
def generate_fleet_data(is_new=False, seed=42):
    global b10_target, pm_target

    np.random.seed(seed)

    # グローバル変数からの設定
    if is_new:
        total_months, n_machines = EVAL_MONTHS_NEW, EVAL_MONTHS_NEW  # 機械は毎月1台ずつ増えると仮定
        b10_target = B10_TARGET_NEW
        pm_target = PM_TARGET_NEW
    else:
        total_months, n_machines = EVAL_MONTHS_OLD, EVAL_MONTHS_OLD  # 機械は毎月1台ずつ増えると仮定
        b10_target = B10_TARGET_OLD
        pm_target = PM_TARGET_OLD

    # ------------------------------------------------
    # 混合ワイブル分布プール (初期・偶発・摩耗) の生成
    # ------------------------------------------------
    POOL_SIZE = 15000  # 十分な量のダミー部品を生成

    # 各部品がどの故障モードの運命を背負うかを発生率で割り当て
    mode_indices = np.random.choice([0, 1, 2], size=POOL_SIZE, p=[m['prob'] for m in WEIBULL_MODES])
    raw_lives = np.zeros(POOL_SIZE)

    # 指定されたβ値を持つ標準ワイブル分布からサンプリング
    for i, m in enumerate(WEIBULL_MODES):
        mask = (mode_indices == i)
        n_mode = np.sum(mask)
        if n_mode > 0:
            raw_lives[mask] = np.random.weibull(m['beta'], n_mode)

    # 混合分布全体のB10を算出し、指定のB10設計ライフになるよう全体をスケーリング
    raw_b10 = np.percentile(raw_lives, 10)
    pool_lives = raw_lives * (b10_target / raw_b10)

    # ------------------------------------------------
    # フリート運用のシミュレーション
    # ------------------------------------------------
    records = []
    pool_idx = 0

    for machine_id in range(1, n_machines + 1):
        start_day = (machine_id - 1) * 30
        end_day = total_months * 30
        current_day = start_day

        while current_day < end_day:
            # プールから部品の真の寿命 (Intrinsic Life) を取り出す
            intrinsic_cycles = pool_lives[pool_idx]
            pool_idx += 1

            # 予防保守(PM)の実施タイミング (目標ライフに対して±5%程度の現場の運用バラツキを付与)
            global pm_cycles
            pm_cycles = np.random.normal(pm_target, pm_target * 0.05)

            # 実際に稼働できたサイクル数 (寿命で壊れるか、予防保守で交換されるか早い方)
            actual_cycles = min(intrinsic_cycles, pm_cycles)
            actual_days = actual_cycles / CYCLES_PER_DAY

            # イベントの判定
            status = 'Failed (事後保守)' if intrinsic_cycles < pm_cycles else 'Censored (予防保守)'

            if current_day + actual_days >= end_day:
                # シミュレーション終了時点でまだ機械の中で動いている場合 (Active)
                observed_days = end_day - current_day
                observed_cycles = observed_days * CYCLES_PER_DAY
                records.append({
                    'Machine_ID': machine_id,
                    'Status': 'Active (稼働中)',
                    'Observed_Cycles': observed_cycles,
                    'Observed_Months': observed_days / 30,
                    'Event_Day': end_day,
                    'Event_Month': total_months
                })
                break
            else:
                # 部品交換 (Failed または Censored) が発生した場合
                current_day += actual_days
                records.append({
                    'Machine_ID': machine_id,
                    'Status': status,
                    'Observed_Cycles': actual_cycles,
                    'Observed_Months': actual_days / 30,
                    'Event_Day': current_day,
                    'Event_Month': int(current_day // 30) + 1
                })

    freet_data_df = pd.DataFrame(records)
    freet_data_df['Month_Bin'] = (freet_data_df['Event_Day'] // 30).astype(int) + 1

    return freet_data_df


# ==========================================
# ブートストラップ用の軽量KM関数 (局所ワイブル回帰 + ブートストラップ)
# ==========================================
def compute_km_simple(durations, events):
    """ブートストラップ用の軽量KM関数"""
    times = np.sort(np.unique(durations))
    survival = []
    n_at_risk = len(durations)
    surv_prob = 1.0
    for t in times:
        deaths = np.sum((durations == t) & events)
        censored = np.sum((durations == t) & ~events)
        if n_at_risk > 0:
            surv_prob *= (1 - deaths / n_at_risk)
        survival.append(surv_prob)
        n_at_risk -= (deaths + censored)
    return np.array(times), np.array(survival)


# ==========================================
# 打ち切り(Censored)データを生かすカプラン・マイヤー(KM)ベースの局所フィット
# ==========================================
def bootstrap_b10_local(df_m, n_boot=200, target_f_limit=0.3):
    b10_estimates = []
    durations = df_m['Observed_Cycles'].values
    events = (df_m['Status'] == 'Failed (事後保守)').values
    n_samples = len(durations)

    # 全体の事後保守が少なすぎる場合 (最低3点は必要)
    if np.sum(events) < 3:
        return np.array([])

    for _ in range(n_boot):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        d_boot = durations[idx]
        e_boot = events[idx]

        t, s = compute_km_simple(d_boot, e_boot)
        f = 1 - s
        mask = (f > 0) & (f <= target_f_limit) & (s > 0)

        # フィットには最低2点のユニークなプロット点が必要
        if np.sum(mask) >= 2:
            X = np.log(t[mask])
            Y = np.log(-np.log(1 - f[mask]))

            # Xに重複しかない(特異行列になる)ケースを排除
            if len(np.unique(X)) >= 2:
                slope, intercept = np.polyfit(X, Y, 1)

                # 傾き(β)に関する発散防止
                # βが極端に小さい(<0.3)とB10が無限大に発散するため、物理的に妥当な傾きのみ採用
                if slope > 0.3:
                    b10_y = np.log(-np.log(0.9))
                    b10_x = (b10_y - intercept) / slope

                    # オーバーフロー防止 (exp(20)は約4.8億サイクル)
                    if b10_x < 20:
                        val = np.exp(b10_x)
                        if not np.isinf(val):
                            b10_estimates.append(val)

    return np.array(b10_estimates)


def fit_weibull_simple(durations, events):
    """全データを用いたシンプルな最小二乗法ワイブル回帰"""
    t, s = compute_km_simple(durations, events)
    mask = (s < 1.0) & (s > 0.0) & (t > 0)
    if np.sum(mask) < 2:
        return None, None
    X = np.log(t[mask])
    Y = np.log(-np.log(s[mask]))
    slope, intercept = np.polyfit(X, Y, 1)
    eta = np.exp(-intercept / slope)
    return slope, eta


def get_weibull_y(f_percent):
    """累積故障率(%)をワイブル軸のY値に変換"""
    return np.log(-np.log(1 - f_percent / 100))


# ==========================================
# KM曲線計算用ヘルパー関数
# ==========================================
def compute_km(durations, events):
    times = np.sort(np.unique(durations))
    survival, ci_lower, ci_upper = [], [], []
    n_at_risk = len(durations)
    surv_prob = 1.0
    var_sum = 0.0

    for t in times:
        deaths = np.sum((durations == t) & events)
        censored = np.sum((durations == t) & ~events)
        if n_at_risk > 0:
            surv_prob *= (1 - deaths / n_at_risk)
            # Greenwoodの公式による分散の累積
            if n_at_risk > deaths:
                var_sum += deaths / (n_at_risk * (n_at_risk - deaths))

        survival.append(float(surv_prob))

        # 95%信頼区間
        se = surv_prob * np.sqrt(var_sum)
        ci_lower.append(max(0.0, surv_prob - 1.96 * se))
        ci_upper.append(min(1.0, surv_prob + 1.96 * se))

        n_at_risk -= (deaths + censored)
    return times, survival, ci_lower, ci_upper


# 色設定
color_active = '#348ABD'    # 青系
color_censored = '#A60628'  # 赤系
color_failed = '#7A68A6'    # 紫系
colors = [color_active, color_censored, color_failed]
labels = ['Active (稼働中)', 'Censored (予防保守)', 'Failed (事後保守)']


def main():
    # ------------------------------------------
    # ダッシュボード描画
    # ------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.4, wspace=0.3)
    fig.suptitle(
        f'早期同等性評価ダッシュボード  (新型 {EVAL_MONTHS_NEW}ヶ月経過 B10設計目標={B10_TARGET_NEW/1000:.0f}k 予防保守目標={PM_TARGET_NEW/1000:.0f}k)',
        fontsize=18, fontweight='bold')

    # データ生成
    df_old = generate_fleet_data(is_new=False)
    df_new = generate_fleet_data(is_new=True)

    print(f'df_new=\n{df_new}')
    print(df_new.groupby(by=['Status'])['Observed_Cycles'].describe())

    # -------------------------------------------------------------------
    # 【上段】 ①② 推移ビュー / ③ バスタブ曲線
    # -------------------------------------------------------------------

    # ① 旧型の現場実績 累積イベント数 (Failed vs Censored)
    if True:
        ax1 = fig.add_subplot(gs[0, 0])
        df_old_plot = df_old[df_old['Status'] != 'Active (稼働中)']    # (Activeを除外し、累積値として描画)

        pivot1 = df_old_plot.groupby(['Month_Bin', 'Status'], observed=False).size().unstack(fill_value=0)
        for col in ['Censored (予防保守)', 'Failed (事後保守)']:
            if col not in pivot1:
                pivot1[col] = 0
        pivot1 = pivot1.reindex(range(1, EVAL_MONTHS_OLD + 1), fill_value=0)
        pivot1_cumsum = pivot1.cumsum()  # 累積和計算

        p1 = ax1.bar(pivot1_cumsum.index, pivot1_cumsum['Censored (予防保守)'], color='skyblue', label='予防保守 (Censored)')
        p2 = ax1.bar(pivot1_cumsum.index, pivot1_cumsum['Failed (事後保守)'], bottom=pivot1_cumsum['Censored (予防保守)'], color='red', label='事後保守 (Failed)')

        # 数値ラベル
        if False:  # 煩雑のため非表示
            fontsize = 6
            ax1.bar_label(p1, label_type='center', fontsize=fontsize, color='black')
            ax1.bar_label(p2, label_type='center', fontsize=fontsize, color='black')

            # 数値ラベル 事後保守+予防保守
            df_new_current = df_new[df_new['Event_Month'] <= EVAL_MONTHS_NEW]
            labels = ['旧型機', '新型機']
            failed_counts = [
                df_old[df_old['Status'] == 'Failed (事後保守)'].shape[0],
                df_new_current[df_new_current['Status'] == 'Failed (事後保守)'].shape[0]]
            censored_counts = [
                df_old[df_old['Status'] == 'Censored (予防保守)'].shape[0],
                df_new_current[df_new_current['Status'] == 'Censored (予防保守)'].shape[0]]
            # ax1.bar_label(p2, labels=[f'Total:{f+c}' for f, c in zip(failed_counts, censored_counts)], padding=3, fontsize=fontsize)

        ax1.set_title('① 旧型の現場実績 (累積部品交換数)')
        ax1.set_xlim(0, EVAL_MONTHS_OLD + 1)
        ax1.set_xlabel('フリート稼働開始からの経過月数')
        ax1.set_ylabel('累積交換数 (個)')

        ax1.legend(loc='upper left')
        # 凡例の並び順を逆転 [Failed, Censored]
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[::-1], labels[::-1])

    # ② 新型の早期評価 (Activeを除外し、累積値として描画)
    if True:
        ax2 = fig.add_subplot(gs[0, 1])
        df_new_plot = df_new[df_new['Status'] != 'Active (稼働中)'].copy()

        pivot2 = df_new_plot.groupby(['Month_Bin', 'Status'], observed=False).size().unstack(fill_value=0)
        for col in ['Censored (予防保守)', 'Failed (事後保守)']:
            if col not in pivot2: pivot2[col] = 0
        pivot2 = pivot2.reindex(range(1, EVAL_MONTHS_NEW + 1), fill_value=0)
        pivot2_cumsum = pivot2.cumsum()  # 累積和計算

        p1 = ax2.bar(pivot2_cumsum.index, pivot2_cumsum['Censored (予防保守)'], color='skyblue', label='予防保守 (Censored)')
        p2 = ax2.bar(pivot2_cumsum.index, pivot2_cumsum['Failed (事後保守)'], bottom=pivot2_cumsum['Censored (予防保守)'], color='red', label='事後保守 (Failed)')

        # 数値ラベル
        fontsize = 6
        ax2.bar_label(p1, label_type='center', fontsize=fontsize, color='black')
        ax2.bar_label(p2, label_type='center', fontsize=fontsize, color='black')

        # 数値ラベル 事後保守+予防保守
        df_new_current = df_new[df_new['Event_Month'] <= EVAL_MONTHS_NEW]
        labels = ['旧型機', '新型機']
        failed_counts = [
            df_old[df_old['Status'] == 'Failed (事後保守)'].shape[0],
            df_new_current[df_new_current['Status'] == 'Failed (事後保守)'].shape[0]]
        censored_counts = [
            df_old[df_old['Status'] == 'Censored (予防保守)'].shape[0],
            df_new_current[df_new_current['Status'] == 'Censored (予防保守)'].shape[0]]
        # ax2.bar_label(p2, labels=[f'Total:{f+c}' for f, c in zip(failed_counts, censored_counts)], padding=3, fontsize=fontsize)
        ax2.set_title('② 新型の早期評価 (累積部品交換数)')

        ax2.set_xlim(0, EVAL_MONTHS_NEW + 1)
        ax2.set_xlabel('フリート稼働開始からの経過月数')
        ax2.set_ylabel('累積交換数 (個)')

        ax2.legend(loc='upper left')
        # 凡例の並び順を逆転 [Failed, Censored]
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles[::-1], labels[::-1])

    # ③ バスタブ曲線
    def draw_bathtub_chart(arg_df):
        print("③バスタブ曲線:")
        df = arg_df[arg_df['Status'] != 'Active (稼働中)']
        df = arg_df[arg_df['Status'] != 'Censored (予防保守)']  # 残りは Failed

        combined_data = np.array(df['Observed_Cycles'].values)
        combined_data.sort()

        # 2. 競合ワイブルモデル（バスタブ型）の関数定義
        def weibull_hazard(t, shape, scale):
            """ワイブル分布の故障率関数 (Hazard Function)"""
            return (shape / scale) * (t / scale)**(shape - 1)

        def bathtub_hazard(t, p1, s1, p2, s2, p3, s3):
            """3つのワイブル分布の和（バスタブ型）"""
            h1 = weibull_hazard(t, p1, s1)  # 初期
            h2 = weibull_hazard(t, p2, s2)  # 偶発
            h3 = weibull_hazard(t, p3, s3)  # 摩耗
            return h1 + h2 + h3

        def log_likelihood(params, data):
            """最尤推定のための対数尤度関数（負の対数尤度を返す）"""
            p1, s1, p2, s2, p3, s3 = params

            # パラメータが正の範囲にあることを制約
            if any(p <= 0 for p in params) or any(s <= 0 for s in params):
                return 1e10

            # 故障率関数 h(t)
            h = bathtub_hazard(data, p1, s1, p2, s2, p3, s3)

            # 累積故障率関数 H(t) (データが小さい順に並んでいると仮定)
            # 簡易的に積分で計算、または数値積分が必要
            # ここでは、データ数が少ない場合の近似的な方法をとるか、
            # 定義に基づいた累積ハザード H(t) = sum( (t/scale)^shape ) を使用

            # 今回は簡単のため、対数尤度を h(t) のみで近似する手法（小データ向け）
            # 正確には survival function S(t) = exp(-H(t)) を使う

            # 累積ハザード H(t) = H1(t) + H2(t) + H3(t)
            H = (data/s1)**p1 + (data/s2)**p2 + (data/s3)**p3

            # 負の対数尤度: - ( Σ log(h(ti)) - Σ H(ti) )
            log_l = np.sum(np.log(h + 1e-10)) - np.sum(H)
            return -log_l

        # 3. バスタブパラメータ推定
        # 初期値
        #  * この指定によって初期・偶発・摩耗の形状に影響する。微妙であり、あまり変えないのが無難。
        #  * α1α2α3は存在しない。このシミュレータではB10目標から逆算しているため
        initial_guess = [
            WEIBULL_MODES[0]['beta'], 1000,    # β1, α1   初期故障
            WEIBULL_MODES[1]['beta'], 40000,   # β2, α2   偶発故障
            WEIBULL_MODES[2]['beta'], 100000,  # β3, α3   摩耗故障
        ]

        # 旧型機において比較的上手くいったときの値
        # initial_guess = [
        #     0.7, 1000,    # β1, α1
        #     1.0, 40000,   # β2, α2
        #     2.5, 100000,  # β3, α3
        # ]

        result = minimize(log_likelihood, initial_guess, args=(combined_data,), method='Nelder-Mead')

        # 4. 結果の出力
        p1, s1, p2, s2, p3, s3 = result.x
        print(f"  推定パラメータ:\n   - 初期: shape={p1:.2f}, scale={s1:.2f}\n   - 偶発: shape={p2:.2f}, scale={s2:.2f}\n   - 摩耗: shape={p3:.2f}, scale={s3:.2f}")

        # 5. 可視化
        # ハザード関数のプロット
        ax3 = fig.add_subplot(gs[0, 2])

        x_vals = np.linspace(0.1, np.max(combined_data), 100)

        # バスタブ曲線プロット
        h_fitted = bathtub_hazard(x_vals, *result.x)
        ax3.plot(x_vals, h_fitted, label='バスタブ')

        # 初期・偶発・摩耗プロット
        d_infant_mortality = Weibull_Distribution(alpha=s1, beta=p1).HF(xvals=x_vals, label='初期', linestyle='--')
        d_random_failures  = Weibull_Distribution(alpha=s2, beta=p2).HF(xvals=x_vals, label='偶発', linestyle='--')
        d_ware_out         = Weibull_Distribution(alpha=s3, beta=p3).HF(xvals=x_vals, label='摩耗', linestyle='--')

        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(cycle_formatter))  # 軸ラベルを k 単位にフォーマット
        ax3.set_xlim(0, B10_TARGET_NEW * CENSORING_FACTOR * 1.2)
        ax3.set_ylim(1e-6, 1e-3)
        ax3.grid(True, which="both", ls="-", alpha=0.2)
        ax3.set_title('③ 旧型のバスタブ曲線 - 故障発生メカニズム')
        ax3.set_xlabel('稼働時間 [サイクル]')
        ax3.set_ylabel('ハザード率 h(t) [log]')
        ax3.set_yscale('log')

        ax3.legend(loc='right', fontsize='x-small')

    draw_bathtub_chart(df_old)

    # -------------------------------------------------------------------
    # 【中段】 ④設計評価 / ⑤運用評価 / ⑥市場生存率
    # -------------------------------------------------------------------

    x_grid = np.linspace(0, B10_TARGET_NEW * CENSORING_FACTOR * 2, 500)

    # ④ 設計評価 (Failed のみ抽出)
    ax4 = fig.add_subplot(gs[1, 0])
    fail_old = df_old[df_old['Status'] == 'Failed (事後保守)']['Observed_Cycles']
    fail_new = df_new[df_new['Status'] == 'Failed (事後保守)']['Observed_Cycles']

    if len(fail_old) > 1:
        ax4.plot(x_grid, stats.gaussian_kde(fail_old)(x_grid), color='orange', label='旧型(事後保守)')
        ax4.fill_between(x_grid, stats.gaussian_kde(fail_old)(x_grid), alpha=0.2, color='orange')
    if len(fail_new) > 1:
        ax4.plot(x_grid, stats.gaussian_kde(fail_new)(x_grid), color='darkblue', label='新型(事後保守)')
        ax4.fill_between(x_grid, stats.gaussian_kde(fail_new)(x_grid), alpha=0.2, color='darkblue')

    # B10ライン (理論値または実績値から算出)
    b10_old_val = np.percentile(df_old[df_old['Status'] == 'Failed (事後保守)']['Observed_Cycles'], 10)
    b10_new_val = np.percentile(df_new[df_new['Status'] == 'Failed (事後保守)']['Observed_Cycles'], 10)
    ax4.axvline(b10_old_val, color='orange', linestyle='--', alpha=1.0)
    ax4.axvline(b10_new_val, color='darkblue', linestyle='--', alpha=1.0)
    ax4.text(b10_old_val, ax4.get_ylim()[1]*0.20, f'旧型\n10% {int(b10_old_val/1000)}k', color='darkblue', ha='left', fontsize=9)
    ax4.text(b10_new_val, ax4.get_ylim()[1]*0.10, f'新型\n10% {int(b10_new_val/1000)}k', color='darkblue', ha='left', fontsize=9)

    ax4.set_title('④ 部品ライフ実力値の比較 - 事後保守(故障)のみ')
    ax4.set_xlabel('稼働時間 [サイクル]')
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(cycle_formatter))  # 軸ラベルを k 単位にフォーマット
    ax4.set_xlim(0, B10_TARGET_NEW * CENSORING_FACTOR * 1.2)
    ax4.legend()

    # ⑤ 運用評価 (Censored のみ抽出)
    ax5 = fig.add_subplot(gs[1, 1])
    cens_old = df_old[df_old['Status'] == 'Censored (予防保守)']['Observed_Cycles']
    cens_new = df_new[df_new['Status'] == 'Censored (予防保守)']['Observed_Cycles']

    if len(cens_old) > 1:
        ax5.plot(x_grid, stats.gaussian_kde(cens_old)(x_grid), color='orange', label='旧型(予防保守)')
        ax5.fill_between(x_grid, stats.gaussian_kde(cens_old)(x_grid), alpha=0.2, color='orange')
    if len(cens_new) > 1:
        ax5.plot(x_grid, stats.gaussian_kde(cens_new)(x_grid), color='blue', label='新型(予防保守)')
        ax5.fill_between(x_grid, stats.gaussian_kde(cens_new)(x_grid), alpha=0.2, color='blue')

    ax5.axvline(PM_TARGET_OLD, color='orange', linestyle=':', linewidth=2)
    ax5.axvline(PM_TARGET_NEW, color='darkblue', linestyle=':', linewidth=2)
    ax5.text(PM_TARGET_OLD, ax5.get_ylim()[1]*0.9, f'旧型 交換目標 {PM_TARGET_OLD/1000:.0f}k', color='darkblue', ha='right')
    ax5.text(PM_TARGET_NEW, ax5.get_ylim()[1]*0.9, f'新型 交換目標 {PM_TARGET_NEW/1000:.0f}k', color='darkblue', ha='left')

    ax5.set_title('⑤ 保守運用の比較 - 予防保守(打ち切り)のみ')
    ax5.set_xlabel('稼働時間 [サイクル]')
    ax5.set_xlim(0, B10_TARGET_NEW * CENSORING_FACTOR * 1.2)
    ax5.xaxis.set_major_formatter(ticker.FuncFormatter(cycle_formatter))  # 軸ラベルを k 単位にフォーマット

    ax5.legend(loc='center left')

    # ⑥ 市場生存率の実績比較と予測 (KM曲線)
    ax6 = fig.add_subplot(gs[1, 2])
    events_old = df_old['Status'] == 'Failed (事後保守)'
    t_old, s_old, l_old, u_old = compute_km(df_old['Observed_Cycles'], events_old)
    ax6.step(t_old, s_old, where='post', color='orange', alpha=0.7, label='旧型 (実績)')
    ax6.fill_between(t_old, l_old, u_old, step='post', color='orange', alpha=0.2)  # 信頼区間

    events_new = df_new['Status'] == 'Failed (事後保守)'
    t_new, s_new, l_new, u_new = compute_km(df_new['Observed_Cycles'], events_new)
    ax6.step(t_new, s_new, where='post', color='darkblue', linewidth=2, label='新型 (実績)')
    ax6.fill_between(t_new, l_new, u_new, step='post', color='darkblue', alpha=0.2)  # 信頼区間

    ax6.axhline(0.9, color='red', linestyle=':', alpha=0.6)  # 90%生存線

    # 交点の算出とプロット (簡易線形補間)
    def draw_b10_marker(times, survival, color, label):
        if min(survival) <= 0.9:
            # 0.9をまたぐ時間を特定
            b10_est = np.interp(0.9, survival[::-1], times[::-1])
            ax6.vlines(b10_est, 0, 0.9, color=color, linestyle='--', alpha=0.7)
            ax6.plot(b10_est, 0.9, 'o', color=color)
            if label == '旧型':
                ha='right'
            elif label == '新型':
                ha='left'
            else:
                assert False
            ax6.text(b10_est, 0.5, f'{label}\n90% {int(b10_est/1000)}k', color=color,
                     ha=ha, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    draw_b10_marker(t_old, s_old, 'orange', '旧型')
    draw_b10_marker(t_new, s_new, 'darkblue', '新型')

    ax6.set_title('⑥ 市場生存率の実績比較 (KM曲線)')
    ax6.set_xlabel('稼働時間 [サイクル]')
    ax6.set_ylabel('生存率')
    ax6.set_ylim(0, 1.05)
    ax6.set_xlim(0, B10_TARGET_NEW * CENSORING_FACTOR * 1.2)
    ax6.xaxis.set_major_formatter(ticker.FuncFormatter(cycle_formatter))  # 軸ラベルを k 単位にフォーマット

    ax6.grid(alpha=0.3)
    ax6.legend()

    # -------------------------------------------------------------------
    # 【下段】 ⑦B10予測(PyMC) / ⑧累積故障数予測 (ワイド表示)
    # -------------------------------------------------------------------

    # ==========================================
    # ワイブル回帰用ヘルパー関数 (カプラン・マイヤー推定量から β, ηを算出)
    # ==========================================
    def fit_weibull_from_km(km_times, km_survival):
        # Pythonのリスト型が渡された場合に備え、NumPy配列に変換する
        km_times = np.array(km_times)
        km_survival = np.array(km_survival)

        # S(t) = 1 または 0 のデータを除外 (log(0)エラー回避)
        mask = (km_survival < 1.0) & (km_survival > 0.0) & (km_times > 0)
        if not np.any(mask):
            return None, None

        # ワイブルプロット用の変数変換: X = ln(t), Y = ln(-ln(S(t)))
        X = np.log(km_times[mask])
        Y = np.log(-np.log(km_survival[mask]))

        if len(X) < 2:
            return None, None

        # 最小二乗法で回帰直線をフィッティング
        slope, intercept = np.polyfit(X, Y, 1)
        beta = slope
        eta = np.exp(-intercept / beta)
        return beta, eta, X, Y

    beta_old, eta_old, X_old, Y_old = fit_weibull_from_km(t_old, s_old)
    beta_new, eta_new, X_new, Y_new = fit_weibull_from_km(t_new, s_new)

    # ------------------------------------------
    # ⑦ B10寿命予測 (ブートストラップ用の軽量KM関数によるCI付き分布)
    # ------------------------------------------

    ax7 = fig.add_subplot(gs[2, 0])

    print('⑦ B10寿命予測')

    # 新型は評価月時点でのデータ(短ライフ側30%)を使用し、ブートストラップ法によりB10ライフを局所フィットで予測する
    df_new_current = df_new[df_new['Event_Month'] <= EVAL_MONTHS_NEW]

    global b10_samples_new
    b10_samples_new = bootstrap_b10_local(df_new_current, n_boot=2000, target_f_limit=0.3)
    print(f' - b10_samples_new={b10_samples_new} ({len(b10_samples_new)})')

    # 旧型は全データを使って精緻な基準値を算出
    b10_samples_old = bootstrap_b10_local(df_old, n_boot=500, target_f_limit=0.3)
    b10_old_baseline = np.median(b10_samples_old) if len(b10_samples_old) > 0 else B10_TARGET_OLD

    if len(b10_samples_new) > 10:
        # KDE描画
        kde = stats.gaussian_kde(b10_samples_new)
        x_vals = np.linspace(min(b10_samples_new) * 0.7, max(b10_samples_new) * 1.3, 500)
        ax7.plot(x_vals, kde(x_vals), color='darkblue', linewidth=2)
        ax7.fill_between(x_vals, 0, kde(x_vals), color='darkblue', alpha=0.2)

        # 同等以上確率の計算と表示
        prob_better = np.mean(b10_samples_new >= b10_old_baseline) * 100
        msg = f"初期故障領域(F<30%)において\n旧型に対する同等以上確率: {prob_better:.1f}%"
        ax7.text(0.4, 0.5, msg, transform=ax7.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        b10_median = np.median(b10_samples_new)
        ci_lower, ci_upper = np.percentile(b10_samples_new, [5, 95])

        ax7.axvline(b10_median, color='darkblue', linestyle='--', label=f'新型 B10推計: {int(b10_median/1000)}k')
        ax7.hlines(y=max(kde(x_vals))*0.05, xmin=ci_lower, xmax=ci_upper, color='black', linewidth=5, label='90% 信頼区間')

    ax7.axvline(b10_old_baseline, color='orange', linestyle='--', linewidth=2, label=f'旧型 B10実績: {int(b10_old_baseline/1000)}k')
    ax7.set_yticks([])
    ax7.set_xlabel('B10寿命 [サイクル]')
    ax7.set_title('⑦ 新型のB10予測分布 (初期故障領域の局所フィット)')
    ax7.legend(loc='upper right', fontsize='x-small')

    ax7.set_xlim(0, B10_TARGET_NEW * CENSORING_FACTOR * 1.2)
    ax7.xaxis.set_major_formatter(ticker.FuncFormatter(cycle_formatter))  # 軸ラベルを k 単位にフォーマット

    # ------------------------------------------
    # グラフ⑧: サチレーション・チャート
    # ------------------------------------------
    ax8 = fig.add_subplot(gs[2, 1])

    months = np.arange(2, EVAL_MONTHS_NEW + 1)
    medians, lowers, uppers, is_local = [], [], [], []

    # 局所フィットと全体フィットの両方を算出
    for m in months:
        df_m = df_new[df_new['Event_Month'] <= m]
        # まず局所フィットを試みる
        samples = bootstrap_b10_local(df_m, n_boot=2000, target_f_limit=0.3)

        if len(samples) > 15:  # 局所フィットに十分なデータがある
            print(f'm={m} 局所フィットに十分なデータがある: len(df_m)={len(df_m)} len(samples)={len(samples)}')

            medians.append(np.median(samples))
            l, u = np.percentile(samples, [5, 95])
            lowers.append(l)
            uppers.append(u)
            is_local.append(True)

        else:  # データ不足時は全体フィットで補完
            print(f'm={m} データ不足時は全体フィットで補完 len(df_m)={len(df_m)} len(samples)={len(samples)}')

            beta_f, eta_f = fit_weibull_simple(df_m['Observed_Cycles'].values, (df_m['Status'] == 'Failed (事後保守)').values)
            if beta_f:
                b10_f = eta_f * (-np.log(0.9))**(1/beta_f)
                medians.append(b10_f)
            else:
                medians.append(np.nan)
            lowers.append(np.nan)
            uppers.append(np.nan)
            is_local.append(False)

    # プロット (スタイルを変えて描画)
    medians = np.array(medians)

    # 新型 (全体フィットが行われた場合のみ表示する)
    num_isnot_local = len([x for x in medians[~np.array(is_local)] if not math.isnan(x)])
    if num_isnot_local >= 1:
        ax8.plot(months[~np.array(is_local)], medians[~np.array(is_local)], 'o--', color='gray', alpha=0.5, label='新型 (全体フィット; 少N期の参考)')

    # 新型 (プロット)
    ax8.plot(months[is_local], medians[is_local], 'o-', color='darkblue', linewidth=2, label=f'新型 (初期故障領域フィット;\n       バンドは90%信頼区間)')
    # 新型 (B10推定値の信頼区間)
    ax8.fill_between(months, lowers, uppers, color='darkblue', alpha=0.15)

    # 新型 (真値)
    ax8.axhline(B10_TARGET_NEW, color='darkblue', alpha=0.3, linestyle='--', label=f'新型 B10目標ライフ {int(B10_TARGET_NEW/1000)}k')
    # 旧型 (実績)
    ax8.axhline(b10_old_baseline, color='orange', linestyle='--', label=f'旧型 B10実績ライフ {int(b10_old_baseline/1000)}k')

    ax8.set_xlim(0, EVAL_MONTHS_NEW)
    ax8.set_title("⑧ 意思決定サチレーション (B10予測の収束)")

    ax8.set_xlabel('稼働経過月数')
    ax8.set_ylabel('B10予測値 [サイクル]')
    ax8.grid(True, alpha=0.4)
    ax8.legend(loc='upper left', fontsize='small')

    ax8.set_xlim(0, EVAL_MONTHS_NEW + 1)
    ax8.set_ylim(0, max([y for y in uppers if not math.isnan(y)]))

    ax8.yaxis.set_major_formatter(ticker.FuncFormatter(cycle_formatter))  # 軸ラベルを k 単位にフォーマット

    # ------------------------------------------
    # ⑨ 局所ワイブルプロット (折れ線フィッティング)
    # ------------------------------------------
    ax9 = fig.add_subplot(gs[2, 2])

    def plot_broken_weibull(ax, df_target, color, label):
        durations = df_target['Observed_Cycles'].values
        events = (df_target['Status'] == 'Failed (事後保守)').values
        t, s = compute_km_simple(durations, events)
        f = 1 - s

        valid_idx = (f > 0) & (s > 0)
        if np.sum(valid_idx) == 0:
            return

        t_valid, f_valid = t[valid_idx], f[valid_idx]

        # 生データのプロット
        X = np.log(t_valid)
        Y = np.log(-np.log(1 - f_valid))
        ax.scatter(X, Y, color=color, alpha=1, s=5)

        # 30%以下 (初期故障モード) + 信頼区間(CI)バンド
        # --------------------------------
        mask1 = (f_valid <= 0.3)
        n_points = np.sum(mask1)

        if n_points > 2:  # CI計算には3点以上必要
            X_m, Y_m = X[mask1], Y[mask1]

            # scipy.stats.linregress を使って回帰と誤差を計算
            res = stats.linregress(X_m, Y_m)
            slope1, intercept1 = res.slope, res.intercept

            # 回帰直線の描画
            ax.plot(X_m, slope1 * X_m + intercept1, color=color, linewidth=1, label=f'{label} (初期故障領域 β={slope1:.1f})')

            # 信頼区間(CI)バンドの描画
            t_val = stats.t.ppf(0.975, n_points - 2)  # 95% CI
            Sxx = np.sum((X_m - np.mean(X_m))**2)

            if Sxx > 0:
                s_err = np.sqrt(np.sum((Y_m - (slope1 * X_m + intercept1))**2) / (n_points - 2))
                # 描画用のX範囲 (少し延長して見栄えを良くする)
                x_band = np.linspace(min(X_m) - 0.1, max(X_m) + 0.1, 50)
                y_hat = slope1 * x_band + intercept1
                ci_width = t_val * s_err * np.sqrt(1/n_points + (x_band - np.mean(X_m))**2 / Sxx)

                # CIの塗りつぶし
                ax.fill_between(x_band, y_hat - ci_width, y_hat + ci_width, color=color, alpha=0.15)

            # B10ラインの描画 (傾きが正の場合のみ)
            if slope1 > 0:
                b10_y = np.log(-np.log(0.9))
                b10_x = (b10_y - intercept1) / slope1
                ax.axhline(y=b10_y, color='red', linestyle=':', alpha=0.5)
                ax.axvline(x=b10_x, color='red', linestyle=':', alpha=0.5)

                if label == '旧型':
                    ax.text(b10_x, get_weibull_y(80), f'旧型 B10={np.exp(b10_x)/1000:.1f}k', color='red', ha='right', fontsize=9, fontweight='bold')
                elif label == '新型':
                    ax.text(b10_x, get_weibull_y(50), f'新型 B10={np.exp(b10_x)/1000:.1f}k', color='red', ha='left', fontsize=9, fontweight='bold')
                else:
                    assert False

        elif n_points == 2:
            # 2点しかない場合はCI計算不能のため直線のみ
            slope1, intercept1 = np.polyfit(X[mask1], Y[mask1], 1)
            ax.plot(X[mask1], slope1 * X[mask1] + intercept1, color=color, linewidth=3, label=f'{label} (初期故障領域 β={slope1:.1f})')

        # 30%以上 (偶発・摩耗モード)
        # --------------------------------
        mask2 = (f_valid > 0.3)
        if np.sum(mask2) >= 2:
            slope2, intercept2 = np.polyfit(X[mask2], Y[mask2], 1)
            ax.plot(X[mask2], slope2 * X[mask2] + intercept2, color=color, linestyle='--', linewidth=1.0, alpha=0.7)

    plot_broken_weibull(ax9, df_old, 'orange', '旧型')
    plot_broken_weibull(ax9, df_new_current, 'darkblue', '新型')

    y_ticks_f = np.array([0.1, 1, 5, 10, 30, 50, 80, 95, 99])  # ワイブルプロットY座標の刻み値 (%)
    ax9.set_yticks(get_weibull_y(y_ticks_f))
    ax9.set_yticklabels([f'{v}%' for v in y_ticks_f])
    ax9.set_ylim(get_weibull_y(0.1), get_weibull_y(99.9))
    ax9.set_title("⑨ 折れ線ワイブル")

    ax9.set_ylabel('累積故障率')
    ax9.set_xlabel('稼働時間 [サイクル]')

    ax9.legend(loc='upper left', fontsize='small')  # fontsize='x-small'

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    # pymc 不具合回避のため、__spec__ がない場合は定義する
    if '__spec__' not in globals():
        __spec__ = None

    main()
