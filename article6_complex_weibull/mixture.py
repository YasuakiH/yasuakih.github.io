# -*- mode:python; -*-

r'''
このスクリプトは、母集団に複数の故障モード（または異なる品質の製品）が混在する場合に
ワイブルプロットやハザード関数、PDF がどのように変化するかを、対話的に理解するためのデモである。

要求:
- 母集団には「良品」と「不良品」が混在していると仮定。
- 各品質群の寿命特性はワイブル分布で表現。
- ユーザは以下を指定可能:
    * B10 ライフ（良品/不良品）
    * ワイブル形状パラメータ β（良品/不良品）
    * 総生産数 N_total
    * 不良品比率 Weight Bad
- スライダー操作でパラメータをリアルタイムに変更可能。

主な機能:
1. B10 ライフと β から各ワイブル分布のスケールパラメータ α を計算。
2. 指定比率で「良品」「不良品」のサンプルを生成。
3. 混合サンプルに基づく可視化:
   - 左上: ハザード関数（混合 + 各群別）
   - 右上: ワイブルプロット（混合サンプル、B10 ライフ表示）
   - 右下: PDF の内訳（混合 + 各群別）
4. ワイブルプロットに B10 ライフを示す垂直線とラベルを表示。
5. サンプル数や比率の変化が可視化に与える影響を学習可能。

学習目的:
- ワイブルプロットが直線にならない場合に、母集団混在の影響を理解。
- 各群の比率や寿命特性の違いがハザードやワイブルプロットにどのように現れるかを視覚的に示す。
- 単一分布解析の誤解を防ぎ、信頼性解析の直感を養う。

実行方法:
JupyterLab で次のコマンドを投入:

%run mixture.py
'''

import numpy as np
import matplotlib.pyplot as plt
from reliability.Distributions import Weibull_Distribution
# from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import Weibull_probability_plot
import ipywidgets as widgets
from ipywidgets import interact  # , VBox


# ---------------------------
# ハザードからサンプリング（補間で右端外挿対応）
# ---------------------------
def sample_from_hazard(t, hazard, n_samples):
    """
    ハザード関数からサンプルを生成する。

    補間によって右端外挿にも対応。

    Parameters
    ----------
    t : ndarray
        時間軸の配列。
    hazard : ndarray
        時間 t におけるハザード関数の値。
    n_samples : int
        生成するサンプル数。

    Returns
    -------
    samples : ndarray
        生成された寿命サンプル。

    Notes
    -----
    各サンプル t_i は、逆累積ハザード法に基づき
    -log(U) = H(t_i) を解くことで生成される。
    """

    dt = t[1] - t[0]
    H = np.cumsum(hazard) * dt
    samples = []
    # H_max = H[- 1]
    for _ in range(n_samples):
        u = np.random.rand()
        target = -np.log(u)
        t_sample = np.interp(target, H, t, left=t[0], right=t[-1] * 1.2)
        samples.append(t_sample)
    return np.array(samples)


# ---------------------------
# B10ライフからα計算
# ---------------------------
def alpha_from_B10(B10, beta):
    """
    B10 ライフからワイブル分布のスケールパラメータ α を計算。

    Parameters
    ----------
    B10 : float
        10%故障ライフ。
    beta : float
        ワイブル形状パラメータ β。

    Returns
    -------
    alpha : float
        ワイブル分布のスケールパラメータ α。

    Notes
    -----
    B10 = α * (-log(0.9))^(1/β) の式を逆算。
    """

    return B10 / (-np.log(0.9))**(1/beta)


# ---------------------------
# メイン描画
# ---------------------------
def simulate(
    B10_good=1200, beta_good=2.0,
    B10_bad=600, beta_bad=0.8,
    N_total=100, w_bad=0.1
):
    """
    混合ワイブルサンプルを生成し、可視化する。

    Parameters
    ----------
    B10_good : float
        良品の B10 ライフ。
    beta_good : float
        良品のワイブル形状パラメータ β。
    B10_bad : float
        不良品の B10 ライフ。
    beta_bad : float
        不良品のワイブル形状パラメータ β。
    N_total : int
        総サンプル数。
    w_bad : float
        不良品の比率 (0～1)。

    Returns
    -------
    None
        matplotlib による図を表示する。

    Notes
    -----
    1. B10とβからαを計算してワイブル分布を定義。
    2. 指定比率に従って良品・不良品サンプルを生成。
    3. 混合サンプルに基づき、以下のグラフを作成:
       - 左上: ハザード関数（混合 + 各群別）
       - 右上: ワイブルプロット（B10表示）
       - 右下: PDF 内訳
    4. w_bad=0 の場合は不良品サンプル生成をスキップ。
    """

    w_good = 1 - w_bad
    alpha_good = alpha_from_B10(B10_good, beta_good)
    alpha_bad = alpha_from_B10(B10_bad, beta_bad)

    # ワイブル定義
    d_good = Weibull_Distribution(alpha=alpha_good, beta=beta_good)
    d_bad  = Weibull_Distribution(alpha=alpha_bad, beta=beta_bad)

    # 時間軸
    t_max = max(B10_good, B10_bad) * 2
    t = np.linspace(0.1, t_max, 10000)

    # ハザードとPDF
    h_good = d_good.HF(t, show_plot=False)
    h_bad  = d_bad.HF(t, show_plot=False)
    pdf_good = d_good.PDF(t, show_plot=False)
    pdf_bad  = d_bad.PDF(t, show_plot=False)

    N_good = int(N_total * w_good)
    N_bad  = N_total - N_good

    samples_good = d_good.random_samples(N_good)
    if N_bad > 0:
        samples_bad = d_bad.random_samples(N_bad)
        samples = np.concatenate([samples_good, samples_bad])
    else:
        samples = samples_good

    np.random.shuffle(samples)

    # 混合
    hazard_mix = w_good * h_good + w_bad * h_bad
    pdf_mix = w_good * pdf_good + w_bad * pdf_bad

    # ---------------------------
    # プロット
    # ---------------------------
    plt.figure(figsize=(14, 8))

    # 左：ハザード
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, hazard_mix, 'k', lw=2.5, label="Mixture Hazard")
    ax1.plot(t, h_good, '--', alpha=0.6, label="Good")
    ax1.plot(t, h_bad, '--', alpha=0.6, label="Bad")
    ax1.set_title("Hazard Function (Mixture)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Hazard")
    ax1.grid()
    ax1.legend()

    # 右上：ワイブルプロット
    ax2 = plt.subplot(2, 2, 2)
    Weibull_probability_plot(failures=samples)
    B10_mixed = np.percentile(samples, 10)
    ax2.axvline(B10_mixed, color='r', linestyle='--')
    ax2.text(B10_mixed * 1.05, 0.2, f"B10={int(B10_mixed)}", color='r')
    ax2.set_xlim(10, 100000)
    ax2.legend(loc='lower left')

    # 右下：PDF内訳
    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(t, pdf_mix, 'k', lw=2.5, label="Mixture PDF")
    ax3.plot(t, w_good * pdf_good, '--', alpha=0.6, label="Good PDF")
    ax3.plot(t, w_bad * pdf_bad, '--', alpha=0.6, label="Bad PDF")
    ax3.set_title("PDF Composition")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("PDF")
    ax3.grid()
    ax3.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------
# UI
# ---------------------------
print(
    '複数の故障モード (品質) が混在する母集団をシミュレートし、ハザード関数とワイブルプロットを描画する\n'
    '　【仮定】部品は摩耗故障型(m=2, B10=1200)であるが、短ライフの不良品(m=0.8、B10=600)が混入する。\n'+
    '1. 初期状態: 良品のみ\n'
    '2. Weight Bad を (1%ずつ) 増加させると不良品が混入する。ワイブルプロットの形状変化を見る。\n'
)

interact(
    simulate,
    B10_good=widgets.FloatSlider(value=1200, min=0, max=2000, step=50, description="B10 Good"),
    beta_good=widgets.FloatSlider(value=2.0, min=0.5, max=3.0, step=0.1, description="β Good"),
    B10_bad=widgets.FloatSlider(value=600, min=0, max=2000, step=50, description="B10 Bad"),
    beta_bad=widgets.FloatSlider(value=0.8, min=0.5, max=3.0, step=0.1, description="β Bad"),
    N_total=widgets.IntSlider(value=1000, min=20, max=5000, step=10, description="Total N"),
    w_bad=widgets.FloatSlider(value=0.0, min=0.0, max=0.5, step=0.01, description="Weight Bad")
)
