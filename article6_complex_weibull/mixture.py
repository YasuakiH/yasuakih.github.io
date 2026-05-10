# -*- mode:python; -*-
# SPDX-License-Identifier: CC0-1.0
"""
mixture.py — 混合ワイブル分布シミュレーター
=============================================

概要
----
製品母集団に「良品」と「不良品」が混在するとき、
ハザード関数・ワイブルプロット・PDF がどのように変化するかを
対話的に確認するデモスクリプト。

想定する学習シナリオ
--------------------
* 良品（摩耗故障型: β > 1）と不良品（初期故障型: β < 1）が混在。
* スライダーで不良品混入率を 0% から徐々に増やしながら
  ワイブルプロットの形状変化を観察する。

主な機能
--------
1. B10 ライフと β から各ワイブル分布のスケールパラメータ α を計算。
2. 指定比率で「良品」「不良品」のサンプルをそれぞれ生成・混合。
3. 以下の 3 グラフをリアルタイム描画:
   - 左上: ハザード関数（混合 + 各群）
   - 右上: ワイブルプロット（B10 マーカー付き）
   - 右下: PDF 内訳（混合 + 各群）

実行方法
--------
JupyterLab で以下を実行::

    %run mixture.py

依存パッケージ
--------------
* numpy
* matplotlib
* reliability
* ipywidgets

作成者: yasuakih (2026)
ライセンス: CC0-1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from reliability.Distributions import Weibull_Distribution
from reliability.Probability_plotting import Weibull_probability_plot
import ipywidgets as widgets
from ipywidgets import interact


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
# ワイブルプロット横軸の表示範囲
WEIBULL_PLOT_XLIM = (10, 100_000)

# 時間軸の解像度（点数）
TIME_AXIS_POINTS = 10_000

# B10 ライフ計算で使う故障確率 10 %
B10_FAILURE_PROBABILITY = 0.10


# ---------------------------------------------------------------------------
# 数学ユーティリティ
# ---------------------------------------------------------------------------
def calc_weibull_scale(b10_life: float, shape: float) -> float:
    """
    B10 ライフとワイブル形状パラメータ β からスケールパラメータ α を計算する。

    ワイブル累積分布関数の定義より::

        F(B10) = 1 - exp(-(B10 / α)^β) = 0.10
        → α = B10 / (-ln(0.90))^(1/β)

    Parameters
    ----------
    b10_life : float
        B10 ライフ（10 % 故障ライフ）。単位は任意（時間・サイクルなど）。
    shape : float
        ワイブル形状パラメータ β（> 0）。

    Returns
    -------
    float
        ワイブルスケールパラメータ α。
    """
    return b10_life / (-np.log(1.0 - B10_FAILURE_PROBABILITY)) ** (1.0 / shape)


# ---------------------------------------------------------------------------
# サンプル生成
# ---------------------------------------------------------------------------
def generate_mixed_samples(
    dist_good: Weibull_Distribution,
    dist_bad: Weibull_Distribution,
    n_total: int,
    fraction_bad: float,
) -> np.ndarray:
    """
    良品・不良品の混合サンプルを生成して返す。

    Parameters
    ----------
    dist_good : Weibull_Distribution
        良品のワイブル分布オブジェクト。
    dist_bad : Weibull_Distribution
        不良品のワイブル分布オブジェクト。
    n_total : int
        生成するサンプルの総数。
    fraction_bad : float
        不良品の混入率 [0, 1]。

    Returns
    -------
    np.ndarray
        ランダム順にシャッフルされた混合寿命サンプル（shape: (n_total,)）。
    """
    n_good = int(n_total * (1.0 - fraction_bad))
    n_bad = n_total - n_good

    samples_good = dist_good.random_samples(n_good)

    if n_bad > 0:
        samples_bad = dist_bad.random_samples(n_bad)
        mixed = np.concatenate([samples_good, samples_bad])
    else:
        mixed = samples_good.copy()

    np.random.shuffle(mixed)
    return mixed


# ---------------------------------------------------------------------------
# 混合分布の統計量計算
# ---------------------------------------------------------------------------
def calc_mixture_statistics(
    time_axis: np.ndarray,
    dist_good: Weibull_Distribution,
    dist_bad: Weibull_Distribution,
    fraction_good: float,
    fraction_bad: float,
) -> dict:
    """
    混合ワイブル分布のハザード関数と PDF を計算して返す。

    混合分布のハザードは単純加重平均ではないが、ここでは
    教育目的のため各群のハザードを比率で重みづけした近似値を用いる。
    PDF は厳密な加重和で計算される。

    Parameters
    ----------
    time_axis : np.ndarray
        計算対象の時間軸配列。
    dist_good : Weibull_Distribution
        良品の分布。
    dist_bad : Weibull_Distribution
        不良品の分布。
    fraction_good : float
        良品の比率 [0, 1]。
    fraction_bad : float
        不良品の比率 [0, 1]。

    Returns
    -------
    dict
        以下のキーを持つ辞書:
        - ``hazard_good``   : 良品ハザード
        - ``hazard_bad``    : 不良品ハザード
        - ``hazard_mixture``: 混合ハザード（近似加重平均）
        - ``pdf_good``      : 良品 PDF
        - ``pdf_bad``       : 不良品 PDF
        - ``pdf_mixture``   : 混合 PDF
    """
    hazard_good = dist_good.HF(time_axis, show_plot=False)
    hazard_bad  = dist_bad.HF(time_axis, show_plot=False)
    pdf_good    = dist_good.PDF(time_axis, show_plot=False)
    pdf_bad     = dist_bad.PDF(time_axis, show_plot=False)

    hazard_mixture = fraction_good * hazard_good + fraction_bad * hazard_bad
    pdf_mixture    = fraction_good * pdf_good    + fraction_bad * pdf_bad

    return {
        "hazard_good":    hazard_good,
        "hazard_bad":     hazard_bad,
        "hazard_mixture": hazard_mixture,
        "pdf_good":       pdf_good,
        "pdf_bad":        pdf_bad,
        "pdf_mixture":    pdf_mixture,
    }


# ---------------------------------------------------------------------------
# 描画ヘルパー
# ---------------------------------------------------------------------------
def plot_hazard_functions(
    ax: plt.Axes,
    time_axis: np.ndarray,
    stats: dict,
) -> None:
    """
    ハザード関数グラフを描画する（左上パネル）。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        描画対象の Axes。
    time_axis : np.ndarray
        時間軸。
    stats : dict
        :func:`calc_mixture_statistics` の戻り値。
    """
    ax.plot(time_axis, stats["hazard_mixture"], "k",  lw=2.5, label="Mixture")
    ax.plot(time_axis, stats["hazard_good"],    "--", alpha=0.6, label="Good")
    ax.plot(time_axis, stats["hazard_bad"],     "--", alpha=0.6, label="Bad")
    ax.set_title("Hazard Function (Mixture)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Hazard rate")
    ax.grid(True)
    ax.legend()


def plot_weibull_probability(
    ax: plt.Axes,
    samples: np.ndarray,
) -> None:
    """
    ワイブルプロットを描画し、B10 ライフ推定値をマーカーで示す（右上パネル）。

    B10 ライフはサンプルの 10 パーセンタイル値で近似する。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        描画対象の Axes。
    samples : np.ndarray
        混合サンプル配列。
    """
    Weibull_probability_plot(failures=samples)

    b10_estimated = np.percentile(samples, 10)
    ax.axvline(b10_estimated, color="r", linestyle="--", lw=1.5)
    ax.text(
        b10_estimated * 1.05,
        0.2,
        f"B10 = {int(b10_estimated)}",
        color="r",
    )
    ax.set_xlim(*WEIBULL_PLOT_XLIM)
    ax.legend(loc="lower left")


def plot_pdf_composition(
    ax: plt.Axes,
    time_axis: np.ndarray,
    stats: dict,
) -> None:
    """
    PDF の内訳グラフを描画する（右下パネル）。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        描画対象の Axes。
    time_axis : np.ndarray
        時間軸。
    stats : dict
        :func:`calc_mixture_statistics` の戻り値。
    """
    ax.plot(time_axis, stats["pdf_mixture"],                  "k",  lw=2.5, label="Mixture PDF")
    ax.plot(time_axis, stats["pdf_good"],                     "--", alpha=0.6, label="Good PDF")
    ax.plot(time_axis, stats["pdf_bad"],                      "--", alpha=0.6, label="Bad PDF")
    ax.set_title("PDF Composition")
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability density")
    ax.grid(True)
    ax.legend()


# ---------------------------------------------------------------------------
# シミュレーション本体（ipywidgets の interact に接続する関数）
# ---------------------------------------------------------------------------
def run_simulation(
    b10_good: float = 1200,
    beta_good: float = 2.0,
    b10_bad: float = 600,
    beta_bad: float = 0.8,
    n_total: int = 1000,
    fraction_bad: float = 0.0,
) -> None:
    """
    混合ワイブルサンプルを生成し、3 種のグラフを描画する。

    ipywidgets の :func:`interact` から呼び出されることを想定している。

    Parameters
    ----------
    b10_good : float
        良品の B10 ライフ（デフォルト: 1200）。
    beta_good : float
        良品のワイブル形状パラメータ β（デフォルト: 2.0、摩耗故障型）。
    b10_bad : float
        不良品の B10 ライフ（デフォルト: 600）。
    beta_bad : float
        不良品のワイブル形状パラメータ β（デフォルト: 0.8、初期故障型）。
    n_total : int
        生成するサンプルの総数（デフォルト: 1000）。
    fraction_bad : float
        不良品の混入率 [0, 1]（デフォルト: 0.0）。

    Notes
    -----
    描画構成 (2×2 グリッド):

    * [1,1] 左上: ハザード関数（空白なし、将来拡張用スペース）
    * [1,2] 右上: ワイブルプロット
    * [2,1] 左下: 未使用
    * [2,2] 右下: PDF 内訳
    """
    fraction_good = 1.0 - fraction_bad

    # --- ワイブル分布オブジェクト生成 ---
    alpha_good = calc_weibull_scale(b10_good, beta_good)
    alpha_bad  = calc_weibull_scale(b10_bad,  beta_bad)
    dist_good  = Weibull_Distribution(alpha=alpha_good, beta=beta_good)
    dist_bad   = Weibull_Distribution(alpha=alpha_bad,  beta=beta_bad)

    # --- 時間軸 ---
    t_max     = max(b10_good, b10_bad) * 2.0
    time_axis = np.linspace(0.1, t_max, TIME_AXIS_POINTS)

    # --- 統計量計算 ---
    stats = calc_mixture_statistics(
        time_axis, dist_good, dist_bad, fraction_good, fraction_bad
    )

    # --- サンプル生成 ---
    samples = generate_mixed_samples(dist_good, dist_bad, n_total, fraction_bad)

    # --- 描画 ---
    fig = plt.figure(figsize=(14, 8))

    ax_hazard  = fig.add_subplot(2, 2, 1)
    ax_weibull = fig.add_subplot(2, 2, 2)
    ax_pdf     = fig.add_subplot(2, 2, 4)

    plot_hazard_functions(ax_hazard, time_axis, stats)
    plot_weibull_probability(ax_weibull, samples)
    plot_pdf_composition(ax_pdf, time_axis, stats)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# エントリーポイント（JupyterLab: %run mixture.py）
# ---------------------------------------------------------------------------
INTRO_MESSAGE = """\
複数の故障モード（品質）が混在する母集団をシミュレートし、ハザード関数とワイブルプロットを描画する

  想定シナリオ
    部品は摩耗故障型（β=2, B10=1200）であるが、
    短ライフの不良品（β=0.8, B10=600）が混入する。

  操作手順
    1. 初期状態: 良品のみ（Weight Bad = 0）
       → ワイブルプロットはほぼ直線（単一ワイブル分布）
    2. Weight Bad を 1% ずつ増加させると不良品が混入する
       → ワイブルプロットが直線からずれ始める形状変化を観察する
"""

print(INTRO_MESSAGE)

interact(
    run_simulation,
    b10_good=widgets.FloatSlider(
        value=1200, min=0, max=2000, step=50,
        description="B10 Good",
        style={"description_width": "120px"},
    ),
    beta_good=widgets.FloatSlider(
        value=2.0, min=0.5, max=3.0, step=0.1,
        description="β Good",
        style={"description_width": "120px"},
    ),
    b10_bad=widgets.FloatSlider(
        value=600, min=0, max=2000, step=50,
        description="B10 Bad",
        style={"description_width": "120px"},
    ),
    beta_bad=widgets.FloatSlider(
        value=0.8, min=0.5, max=3.0, step=0.1,
        description="β Bad",
        style={"description_width": "120px"},
    ),
    n_total=widgets.IntSlider(
        value=1000, min=20, max=5000, step=10,
        description="Total N",
        style={"description_width": "120px"},
    ),
    fraction_bad=widgets.FloatSlider(
        value=0.0, min=0.0, max=0.5, step=0.01,
        description="Weight Bad",
        style={"description_width": "120px"},
    ),
)
