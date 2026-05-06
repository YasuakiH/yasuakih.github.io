import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy.special import gamma
from lifelines import KaplanMeierFitter
from reliability.Probability_plotting import Weibull_probability_plot
from reliability.Distributions import Weibull_Distribution
import warnings

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def weibull_axis_value(cumulative_failure_percent: float) -> float:
    """累積故障率(%) をワイブル確率紙の Y 軸値 ln(-ln(1-F)) に変換する。"""
    return np.log(-np.log(1 - cumulative_failure_percent / 100))


# ---------------------------------------------------------------------------
# 1. シミュレーション・エンジン
# ---------------------------------------------------------------------------

# replacement_cause (部品交換の原因コード)
CAUSE_PREVENTIVE = 0   # 予防保守(定期点検による打ち切り)
CAUSE_FAILURE    = 1   # 寿命到達による故障交換


class ReliabilitySimulator:
    """
    フリート内の部品ライフをシミュレートする。

    各機械は 1 個の部品を搭載し、故障または予防保守のタイミングで交換される。
    不良品混入開始日 T1 (contamination_day) 以降の交換では、mix_rate_fraction の確率で不良品が混入する。
    """

    def __init__(self, cfg, mix_rate_percent: float):
        self.cfg = cfg
        self.mix_rate_fraction = mix_rate_percent / 100.0
        self._run_simulation()

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _new_machine(self, machine_id: int, start_day: int, is_defective: bool) -> dict:
        """機械(部品搭載状態)の初期辞書を生成する。"""
        alpha = self.cfg.ALPHA_BAD  if is_defective else self.cfg.ALPHA_GOOD
        beta  = self.cfg.BETA_BAD   if is_defective else self.cfg.BETA_GOOD
        return {
            'id':           machine_id,
            'start_day':    start_day,
            'is_defective': is_defective,
            'part_life_cycles': max(np.random.weibull(beta) * alpha, 1.0),
        }

    def _draw_replacement_part(self, today: int) -> bool:
        """今日の交換部品が不良品かどうかを確率的に決定する。"""
        contamination_active = (today >= self.cfg.CONTAMINATION_DAY)
        return contamination_active and (np.random.rand() < self.mix_rate_fraction)

    # ------------------------------------------------------------------
    # シミュレーション本体
    # ------------------------------------------------------------------

    def _run_simulation(self):
        cfg = self.cfg
        machine_id_counter = 0
        fleet: list[dict] = []
        replacement_events: list[dict] = []
        daily_stats: list[dict] = []

        # フリートの初期機械(T=1 から稼働)
        fleet.append(self._new_machine(machine_id_counter, start_day=1, is_defective=False))
        machine_id_counter += 1

        for today in range(1, cfg.TOTAL_DAYS + 1):

            # フリート増強(RAMP_UP_INTERVAL ごとに 1 台追加)
            if today > 1 and today % cfg.RAMP_UP_INTERVAL == 1:
                fleet.append(self._new_machine(machine_id_counter, start_day=today, is_defective=False))
                machine_id_counter += 1

            today_good_replacements      = 0
            today_defective_replacements = 0
            today_good_operating      = 0
            today_defective_operating = 0

            for machine in fleet:
                part_age_cycles = (today - machine['start_day']) * cfg.CYCLES_PER_DAY

                # 交換判定
                replacement_cause = None
                if part_age_cycles >= machine['part_life_cycles']:
                    # 寿命到達 → 故障交換
                    replacement_cause = CAUSE_FAILURE
                elif today % cfg.INSPECTION_INTERVAL == 0:
                    # 定期点検: 次回点検前に TARGET_B10 を超えそうなら予防交換
                    projected_age = part_age_cycles + cfg.INSPECTION_INTERVAL * cfg.CYCLES_PER_DAY
                    if projected_age > cfg.TARGET_B10:
                        replacement_cause = CAUSE_PREVENTIVE

                if replacement_cause is not None:
                    # 交換件数カウント
                    if machine['is_defective']:
                        today_defective_replacements += 1
                    else:
                        today_good_replacements += 1

                    # イベント記録(打ち切り or 故障)
                    recorded_duration = (
                        part_age_cycles if replacement_cause == CAUSE_PREVENTIVE
                        else machine['part_life_cycles']
                    )
                    replacement_events.append({
                        'duration':     max(recorded_duration, 1.0),
                        'event':        replacement_cause,      # 0=打ち切り, 1=故障
                        'is_defective': machine['is_defective'],
                        'day':          today,
                    })

                    # 交換後の新部品を装着
                    new_is_defective = self._draw_replacement_part(today)
                    machine.update(self._new_machine(
                        machine['id'],
                        start_day=today,
                        is_defective=new_is_defective,
                    ))

                # 稼働台数カウント(交換後の状態で集計)
                if machine['is_defective']:
                    today_defective_operating += 1
                else:
                    today_good_operating += 1

            daily_stats.append({
                'good_operating':      today_good_operating,
                'defective_operating': today_defective_operating,
                'good_replacements':      today_good_replacements,
                'defective_replacements': today_defective_replacements,
            })

        self.df_events = pd.DataFrame(replacement_events)
        self.df_daily  = pd.DataFrame(daily_stats)


# ---------------------------------------------------------------------------
# 設定クラス
# ---------------------------------------------------------------------------

class Config:
    # --- 部品・機械の物理パラメータ ---
    TARGET_B10             = 800_000   # 設計寿命(B10 寿命)[サイクル]
    CYCLES_PER_DAY         = 20_000    # 1 機械あたりの 1 日稼働サイクル数
    INSPECTION_INTERVAL    = 14        # 定期点検の間隔 [日]
    RAMP_UP_INTERVAL       = 30        # フリート増強の間隔(30 日ごとに +1 台)[日]
    TOTAL_DAYS             = 365       # シミュレーション期間 [日]

    # --- ワイブル形状パラメータ ---
    BETA_GOOD = 1.5   # 良品: 摩耗故障モード (β > 1)
    BETA_BAD  = 0.8   # 不良品: 初期故障モード (β < 1)

    # --- スケールパラメータ(Config.update() で自動計算) ---
    ALPHA_GOOD = 0.0
    ALPHA_BAD  = 0.0

    # --- 不良品混入タイミング(デフォルト値) ---
    CONTAMINATION_DAY = 181   # この日以降の交換から不良品が混入し始める

    # --- 年間想定交換件数(α 逆算に使用) ---
    EXPECTED_REPLACEMENTS_PER_YEAR = 50

    # --- CUSUM ベースライン推定 ---
    # T=0からCUSUM_BASELINE_WINDOW_DAYS 日間をベースライン期間とする
    # 留意点: T2 が T1(不良品混入開始日)に近い場合、ウィンドウが混入後のデータをまたぐ可能性がある(ベースライン汚染)。
    CUSUM_BASELINE_WINDOW_DAYS = 60   # T2 から遡るベースライン期間 [日]

    # --- 描画カラー ---
    COLOR_GOOD      = '#558B2F'   # 良品: 緑
    COLOR_DEFECTIVE = '#D84315'   # 不良品: 赤

    @classmethod
    def update(cls):
        """
        ALPHA_GOOD / ALPHA_BAD をシミュレーション条件から逆算する。
        - ALPHA_GOOD: 年間交換件数の目標から MTTF を逆算
        - ALPHA_BAD : 良品 B10 の 25% を不良品 B10 として設定
        """
        avg_machine_count       = (cls.TOTAL_DAYS / cls.RAMP_UP_INTERVAL) / 2
        mean_cycles_to_failure  = (
            cls.TOTAL_DAYS * cls.CYCLES_PER_DAY * avg_machine_count
            / cls.EXPECTED_REPLACEMENTS_PER_YEAR
        )
        cls.ALPHA_GOOD = mean_cycles_to_failure / gamma(1 + 1 / cls.BETA_GOOD)

        good_b10_cycles     = cls.ALPHA_GOOD * (-np.log(0.9)) ** (1 / cls.BETA_GOOD)
        defective_b10_cycles = good_b10_cycles * 0.25
        cls.ALPHA_BAD = defective_b10_cycles / (-np.log(0.9)) ** (1 / cls.BETA_BAD)


np.random.seed(42)
Config.update()


# ---------------------------------------------------------------------------
# 2. 対話型ダッシュボード
# ---------------------------------------------------------------------------

def update_dashboard(baseline_day: int, evaluation_day: int, mix_rate_percent: int):
    """
    スライダー値を受け取り、シミュレーションを実行して 7 つのチャートを描画する。

    Parameters
    ----------
    baseline_day     : T1: 不良品混入開始日
                       ※ CUSUM のベースラインは Config.CUSUM_BASELINE_WINDOW_DAYS に固定で、T1へ依存しない。
    evaluation_day   : T2: ワイブル・KM・ヒストグラムの評価日
    mix_rate_percent : 不良品混入率 [%]
    """

    np.random.seed(42)  # スライダー操作にかかわらず、標本を一定にしたい

    simulator = ReliabilitySimulator(Config, mix_rate_percent)
    fig = plt.figure(figsize=(20, 10))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])

    timeline_days = np.arange(1, Config.TOTAL_DAYS + 1)

    # ------------------------------------------------------------------
    # ① フリート稼働・交換状況
    # ------------------------------------------------------------------
    ax_fleet = fig.add_subplot(grid[0, 0])
    ax_fleet_r = ax_fleet.twinx()

    ax_fleet.bar(
        timeline_days, simulator.df_daily['good_operating'],
        color='#AED581', alpha=0.4, label='稼働数 (良品付き)',
    )
    ax_fleet.bar(
        timeline_days, simulator.df_daily['defective_operating'],
        bottom=simulator.df_daily['good_operating'],
        color='#FF8A65', alpha=0.4, label='稼働数 (不良品付き)',
    )
    ax_fleet_r.step(
        timeline_days, simulator.df_daily['good_replacements'].cumsum(),
        color=Config.COLOR_GOOD, lw=1.5, label='累計交換 (良品)',
    )
    ax_fleet_r.step(
        timeline_days, simulator.df_daily['defective_replacements'].cumsum(),
        color=Config.COLOR_DEFECTIVE, lw=1.5, label='累計交換 (不良品)',
    )
    ax_fleet.set_title("① フリート稼働・交換状況", fontsize=10)
    ax_fleet.set_xlim(0, Config.TOTAL_DAYS)
    ax_fleet.set_xlabel("経過日数", fontsize=9)
    ax_fleet.set_ylabel("機械の稼働数", fontsize=9)
    ax_fleet_r.set_ylabel("部品交換数 (累積)", fontsize=9)
    ax_fleet.legend(loc='upper left', fontsize='small')

    # ------------------------------------------------------------------
    # ② 故障発生タイミング
    # ------------------------------------------------------------------
    ax_scatter = fig.add_subplot(grid[0, 1])

    scatter_specs = [
        # (is_defective, cause, label, marker, color)
        (False, CAUSE_FAILURE, '良品',   'x', Config.COLOR_GOOD),
        (True,  CAUSE_FAILURE, '不良品', 'x', Config.COLOR_DEFECTIVE),
    ]
    for is_defective, cause, label, marker, color in scatter_specs:
        subset = simulator.df_events[
            (simulator.df_events['is_defective'] == is_defective)
            & (simulator.df_events['event'] == cause)
        ]
        ax_scatter.scatter(
            subset['day'], subset['duration'] / 1_000,
            label=label, marker=marker, color=color, alpha=0.6, s=25,
        )
    ax_scatter.set_title("② 故障発生タイミング", fontsize=10)
    ax_scatter.set_xlim(0, Config.TOTAL_DAYS)
    ax_scatter.set_xlabel("経過日数", fontsize=9)
    ax_scatter.set_ylabel("Cycles [k]", fontsize=9)
    ax_scatter.legend(fontsize='small')

    # ------------------------------------------------------------------
    # ③ ワイブル確率紙
    # ------------------------------------------------------------------
    ax_weibull = fig.add_subplot(grid[1, 0])
    plt.sca(ax_weibull)

    for target_day, line_color in [(baseline_day, 'blue'), (evaluation_day, 'red')]:
        df_up_to_day = simulator.df_events[simulator.df_events['day'] <= target_day]
        failures  = df_up_to_day[df_up_to_day['event'] == CAUSE_FAILURE  ]['duration'].values
        censored  = df_up_to_day[df_up_to_day['event'] == CAUSE_PREVENTIVE]['duration'].values
        if len(failures) >= 2:
            Weibull_probability_plot(
                failures=failures, right_censored=censored,
                color=line_color, alpha=0.2, label=f'T={target_day}',
            )
    ax_weibull.set_title("③ 統計的解析 (ワイブル)", fontsize=10)
    ax_weibull.set_xlim(0, Config.TARGET_B10 * 2)
    ax_weibull.set_xlabel('Cycles', fontsize=9)
    ax_weibull.set_ylabel('故障率', fontsize=9)
    ax_weibull.legend(fontsize='small')

    # ------------------------------------------------------------------
    # ④ KM プロット(生存曲線)
    # ------------------------------------------------------------------
    ax_km = fig.add_subplot(grid[1, 1])
    kmf = KaplanMeierFitter()

    for target_day, line_color in [(baseline_day, 'blue'), (evaluation_day, 'red')]:
        df_up_to_day = simulator.df_events[simulator.df_events['day'] <= target_day]
        if not df_up_to_day.empty:
            kmf.fit(df_up_to_day['duration'], df_up_to_day['event'], label=f"T={target_day}")
            kmf.plot_survival_function(ax=ax_km, color=line_color, ci_show=True)

    ax_km.set_title("④ 生存曲線 (KMプロット)", fontsize=10)
    ax_km.set_xlabel('Cycles', fontsize=9)
    ax_km.set_ylabel('生存率', fontsize=9)
    ax_km.set_ylim(0, 1)
    ax_km.legend(fontsize='small')

    # ------------------------------------------------------------------
    # ⑤ CUSUM 異常検知
    # ------------------------------------------------------------------
    ax_cusum   = fig.add_subplot(grid[2, :2])
    ax_cusum_r = ax_cusum.twinx()   # 右軸: 故障イベント表示用

    df_cusum        = simulator.df_daily.iloc[:evaluation_day].copy()
    cusum_days      = np.arange(1, len(df_cusum) + 1)
    active_machines = df_cusum['good_operating'] + df_cusum['defective_operating']
    daily_failures  = df_cusum['good_replacements'] + df_cusum['defective_replacements']

    # ベースライン故障率の推定
    # --------------------------------
    # ベースラインは冒頭固定方式とし、経過日数 T=0 から CUSUM_BASELINE_WINDOW_DAYS 日間をベースライン期間とする。
    # * メリット  : ベースライン汚染が起こる可能性を引き下げる (スライディングウィンドウと比較して)。
    # * デメリット: シミュレーション冒頭はサンプル数が少なく、ベースライン推定の精度が低下する。
    # * ベースライン汚染: ベースライン期間のウィンドウが不良品混入をまたぐと、不良品の故障データがベースラインに混入し、管理限界が高くなりすぎて検出遅れにつながる。
    # --------------------------------

    baseline_window_start = 0                                                       # ベースライン期間の起点
    baseline_window_end   = min(Config.CUSUM_BASELINE_WINDOW_DAYS, evaluation_day)  # ベースライン期間の終点

    baseline_mask = (cusum_days >= baseline_window_start) & (cusum_days <= baseline_window_end)

    baseline_failure_rate = (
        daily_failures[baseline_mask].sum() / active_machines[baseline_mask].sum()
        if active_machines[baseline_mask].sum() > 0 else 0.0
    )

    # ベースラインからの日次偏差
    deviations   = daily_failures - (active_machines * baseline_failure_rate)
    baseline_std = np.std(deviations[baseline_mask])
    baseline_std = baseline_std if baseline_std > 0 else 1e-5

    cusum_slack     = 2.0 * baseline_std    # 許容変化量 k(検出感度調整)
    cusum_threshold = 12.0 * baseline_std   # 管理限界 H(アラート閾値)

    # CUSUM 累積値の計算(片側上方 CUSUM)
    cusum_values = np.zeros(len(df_cusum))
    for i in range(1, len(df_cusum)):
        cusum_values[i] = max(0.0, cusum_values[i - 1] + deviations.iloc[i] - cusum_slack)

    # --- 左軸: CUSUM 折れ線と管理限界 ---
    ax_cusum.plot(cusum_days, cusum_values, color='purple', lw=2.5, label='CUSUM', zorder=3)
    ax_cusum.axhline(cusum_threshold, color='red', ls='--',
                     label=f'管理限界 H = {cusum_threshold:.2f}')

    # ベースライン推定ウィンドウの網掛け表示
    ax_cusum.axvspan(baseline_window_start, baseline_window_end,
                     color='steelblue', alpha=0.08,
                     label=f'ベースライン推定期間 (T=0～{Config.CUSUM_BASELINE_WINDOW_DAYS})')
    ax_cusum.axvline(baseline_window_start, color='steelblue', ls=':', lw=1.2)

    # --- 右軸: 故障イベントを縦線で重ね描き(良品・不良品を色分け) ---
    df_events_cusum       = simulator.df_events[simulator.df_events['day'] <= evaluation_day]
    df_good_failures      = df_events_cusum[
        (df_events_cusum['event'] == CAUSE_FAILURE) & (df_events_cusum['is_defective'] == False)
    ]
    df_defective_failures = df_events_cusum[
        (df_events_cusum['event'] == CAUSE_FAILURE) & (df_events_cusum['is_defective'] == True)
    ]

    # 良品故障: 短い緑色の縦実線
    ax_cusum_r.vlines(
        df_good_failures['day'], ymin=-0.8, ymax=0.2,
        color=Config.COLOR_GOOD, alpha=1.0, lw=1.2, label='故障 (良品)',
    )
    # 不良品故障: 長い赤色の縦実線
    ax_cusum_r.vlines(
        df_defective_failures['day'], ymin=-0.4, ymax=0.4,
        color=Config.COLOR_DEFECTIVE, alpha=1.0, lw=1.5, label='故障 (不良品)',
    )

    ymax = 6  # Y軸レンジ: -1 - ymax
    # 不良品混入開始: 長い赤色の縦破線
    ax_cusum_r.vlines(
        baseline_day, ymin=-0.4, ymax=ymax,
        color='gray', alpha=1.0, lw=1.5, ls=':', label='不良品混入開始',
    )
    ax_cusum.text(baseline_day, ymax / 2, f'不良品混入開始\nT1={baseline_day}', color='black', ha='right', fontsize='small')

    ax_cusum_r.set_ylim(0, 4)          # 縦線が主軸と干渉しないよう余白を確保
    ax_cusum_r.set_ylabel("故障イベント", fontsize=9)
    ax_cusum_r.set_yticks([])          # 右軸の目盛り数値は不要

    # --- アラート注記 ---
    alert_mask = cusum_values > cusum_threshold
    if np.any(alert_mask):
        alert_day_index = np.where(alert_mask)[0][0]
        ax_cusum.annotate(
            f"異常検知 T={cusum_days[alert_day_index]}",
            xy=(cusum_days[alert_day_index], cusum_values[alert_day_index]),
            xytext=(cusum_days[alert_day_index] - 60, 5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1),
            fontsize='small',
            zorder=5,
        )
        # ax_cusum.set_facecolor('mistyrose')

    ax_cusum.set_ylim(-1, ymax)
    ax_cusum.set_xlim(1, evaluation_day)
    ax_cusum.set_title("⑤ 異常検知アラート (CUSUM法) ", fontsize=10)
    ax_cusum.set_xlabel("経過日数", fontsize=9)
    ax_cusum.set_ylabel("CUSUM 値", fontsize=9)

    # 左軸・右軸の凡例を統合して表示
    handles_l, labels_l = ax_cusum.get_legend_handles_labels()
    handles_r, labels_r = ax_cusum_r.get_legend_handles_labels()
    ax_cusum.legend(handles_l + handles_r, labels_l + labels_r,
                    loc='upper left', fontsize='small')

    # ------------------------------------------------------------------
    # ⑥ 故障分布ヒストグラム (T1・T2 それぞれ)
    # ------------------------------------------------------------------
    histogram_bins = np.linspace(0, 1.0e6, 11)

    for row_index, target_day in enumerate([baseline_day, evaluation_day]):
        ax_hist = fig.add_subplot(grid[row_index, 2])
        df_failures = simulator.df_events[
            (simulator.df_events['day'] <= target_day)
            & (simulator.df_events['event'] == CAUSE_FAILURE)
        ]
        good_durations      = df_failures[df_failures['is_defective'] == False]['duration']
        defective_durations = df_failures[df_failures['is_defective'] == True ]['duration']

        ax_hist.hist(
            [good_durations, defective_durations],
            bins=histogram_bins,
            stacked=True,
            color=[Config.COLOR_GOOD, Config.COLOR_DEFECTIVE],
            label=['良品', '不良品'],
            alpha=0.8,
        )
        ax_hist.set_title(f"⑥ 故障分布 (T={target_day})", fontsize=10)
        ax_hist.set_ylim(0, 20)
        ax_hist.set_xlabel('Cycles', fontsize='small')
        ax_hist.set_ylabel('交換部品数', fontsize='small')
        ax_hist.legend(fontsize='small')

    # ------------------------------------------------------------------
    # ⑦ ハザード関数 (物理的特性)
    # ------------------------------------------------------------------
    ax_hazard = fig.add_subplot(grid[2, 2])

    dist_good      = Weibull_Distribution(alpha=Config.ALPHA_GOOD, beta=Config.BETA_GOOD)
    dist_defective = Weibull_Distribution(alpha=Config.ALPHA_BAD,  beta=Config.BETA_BAD)
    cycles_range   = np.linspace(1, 1.5e6, 200)

    hazard_good      = dist_good.HF(cycles_range)
    hazard_defective = dist_defective.HF(cycles_range)

    ax_hazard.plot(cycles_range, hazard_good,      color=Config.COLOR_GOOD,      lw=2, label=f'良品 (m値={Config.BETA_GOOD})')
    ax_hazard.plot(cycles_range, hazard_defective, color=Config.COLOR_DEFECTIVE, lw=2, label=f'不良品 (m値={Config.BETA_BAD})')
    ax_hazard.set_title("⑦ ハザード関数 (物理的特性)", fontsize=10)
    ax_hazard.set_yscale('log')
    ax_hazard.set_xlabel("Cycles", fontsize=9)
    ax_hazard.set_ylabel("ハザード関数 h(t)", fontsize=9)
    ax_hazard.legend(fontsize='small')
    ax_hazard.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. インタラクティブ UI
# ---------------------------------------------------------------------------

output_area = widgets.Output()


def on_slider_change(change):
    with output_area:
        clear_output(wait=True)
        update_dashboard(
            baseline_day    = baseline_day_slider.value,
            evaluation_day  = evaluation_day_slider.value,
            mix_rate_percent = mix_rate_slider.value,
        )


baseline_day_slider = widgets.IntSlider(
    value=180, min=30, max=360, description='T1 (不良品混入開始)',
    step=30,
    style={'description_width': 'initial'},
)
evaluation_day_slider = widgets.IntSlider(
    value=180, min=30, max=360, description='T2 (③④⑤⑥評価)',
    step=30,
    style={'description_width': 'initial'}
)
mix_rate_slider = widgets.IntSlider(
    value=30, min=0, max=100, step=10, description='不良率%',
)

for slider in [baseline_day_slider, evaluation_day_slider, mix_rate_slider]:
    slider.observe(on_slider_change, names='value')

display(widgets.VBox([
    widgets.HBox([baseline_day_slider, evaluation_day_slider, mix_rate_slider]),
    output_area,
]))
on_slider_change(None)
