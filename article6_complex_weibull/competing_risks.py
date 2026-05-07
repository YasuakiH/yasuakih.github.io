# -*- mode:python; -*-
r'''
競合ワイブルサンプル可視化デモ（初期故障・偶発故障・摩耗故障）

目的:
本スクリプトは、典型的なバスタブ型ハザードを持つ製品寿命のモデルを、
ワイブル分布の競合としてシミュレーションし、可視化するデモである。
ユーザは初期故障、偶発故障、摩耗故障の寄与率を調整でき、
競合ハザード関数とワイブルプロット上でのB10ライフを確認できる。

要求:
- 製品寿命は3つの故障モード（初期故障・偶発故障・摩耗故障）で表現される。
- 各故障モードはワイブル分布でモデル化される。
- ユーザは各故障モードのウェイトを指定し、また全体のサンプル数を調整できる。
- 競合ハザード関数、PDFからB10ライフを算出し、ワイブルプロット上に表示する。

主な機能:
1. 各モードのハザード関数・PDF・生存関数を計算。
2. 指定されたウェイトに従い、競合サンプルを生成。
3. 左側にハザード関数、右側にワイブルプロットを表示。
4. ワイブルプロット上に競合ハザードから推定したB10ライフを表示。
5. ipywidgets により、ウェイトやサンプル数をリアルタイムに変更可能。

学習目的:
- ワイブルプロットが直線にならない場合に、競合故障モードの存在を理解。
- 初期故障や摩耗故障がハザード関数やワイブルプロットに与える影響を視覚化。
- B10ライフとハザードの関係を直感的に把握。

実行方法
JupyterLab で次のコマンドを投入:

%run competing_risks.py
'''

import numpy as np
import matplotlib.pyplot as plt

from reliability.Distributions import Weibull_Distribution
from reliability.Probability_plotting import Weibull_probability_plot

import ipywidgets as widgets
from ipywidgets import interact

# =================================================
# B10ライフをハザード関数から計算する補助関数
# =================================================
def compute_b10(t, h_mix):
    """
    ハザード関数 h_mix(t) からB10ライフを推定する。

    Parameters
    ----------
    t : ndarray
        時間軸
    h_mix : ndarray
        競合ハザード関数

    Returns
    -------
    float
        推定B10ライフ
    """
    dt = t[1] - t[0]

    # CDF = 1 - exp(-∫h dt)
    H = np.cumsum(h_mix) * dt
    F = 1 - np.exp(-H)

    # F=0.1に最も近い時間をB10として返す
    idx = np.argmin(np.abs(F - 0.1))
    return t[idx]

# =================================================
# シミュレーション本体
# =================================================
def simulate(w_early=0.0, w_wear=0.0, n=1500):
    """
    初期故障・偶発故障・摩耗故障の競合ハザードをシミュレーションし、
    ハザード関数とワイブルプロットを描画する。

    Parameters
    ----------
    w_early : float
        初期故障のウェイト（0~0.5推奨）
    w_wear : float
        摩耗故障のウェイト（0~0.5推奨）
    n : int
        サンプル数
    """

    # ---------------------------
    # 各故障モードのワイブル分布を定義
    # ---------------------------
    d_early  = Weibull_Distribution(alpha=50,  beta=0.5)  # 初期故障
    d_random = Weibull_Distribution(alpha=200, beta=1.0)  # 偶発故障（ベース）
    d_wear   = Weibull_Distribution(alpha=500, beta=3.5)  # 摩耗故障

    # ---------------------------
    # ウェイトの正規化
    # ---------------------------
    w_random = 1 - w_early - w_wear
    w_random = max(w_random, 0)
    weights = np.array([w_early, w_random, w_wear])
    weights = weights / weights.sum()

    # ---------------------------
    # 時間軸
    # ---------------------------
    t = np.linspace(0.1, 1000, 2000)

    # ---------------------------
    # ハザード、PDF、生存関数の計算
    # ---------------------------
    h_early  = d_early.HF(t, show_plot=False)
    h_random = d_random.HF(t, show_plot=False)
    h_wear   = d_wear.HF(t, show_plot=False)

    pdf = (
        weights[0] * d_early.PDF(t, show_plot=False) +
        weights[1] * d_random.PDF(t, show_plot=False) +
        weights[2] * d_wear.PDF(t, show_plot=False)
    )

    sf = (
        weights[0] * d_early.SF(t, show_plot=False) +
        weights[1] * d_random.SF(t, show_plot=False) +
        weights[2] * d_wear.SF(t, show_plot=False)
    )

    h_mix = pdf / sf

    # ---------------------------
    # 競合サンプルの生成（モード選択型）
    # ---------------------------
    samples = []
    for _ in range(n):
        r = np.random.rand()
        if r < weights[0]:
            samples.append(d_early.random_samples(1)[0])
        elif r < weights[0] + weights[1]:
            samples.append(d_random.random_samples(1)[0])
        else:
            samples.append(d_wear.random_samples(1)[0])
    samples = np.array(samples)

    # =================================================
    # 描画
    # =================================================

    # ---------------------------
    # 左：ハザード関数
    # ---------------------------
    plt.subplot(121)
    ax1 = plt.gca()
    ax1.plot(t, h_mix, 'k', lw=2.5, label="Mixture Hazard")
    ax1.plot(t, h_early, '--', alpha=0.6, label="Early")
    ax1.plot(t, h_random, '--', alpha=0.6, label="Random")
    ax1.plot(t, h_wear, '--', alpha=0.6, label="Wear-out")
    ax1.set_title("Hazard Function (Physical Model)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Hazard")
    ax1.grid()
    ax1.legend()

    # ---------------------------
    # 右：ワイブルプロット
    # ---------------------------
    plt.subplot(122)
    ax2 = plt.gca()
    Weibull_probability_plot(failures=samples)
    ax2.set_xlim(0.01, 10000)

    # B10ライフを表示
    est_b10 = compute_b10(t, h_mix)
    ax2.axvline(est_b10, color='red', linestyle='--', lw=2)
    ax2.text(
        est_b10,
        0.5,
        f"B10 = {est_b10:.1f}",
        rotation=90,
        color='red',
        va='center',
        ha='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # 図の調整
    plt.gcf().set_size_inches(13, 6)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.91, wspace=0.30)
    plt.show()


# =================================================
# UI（ipywidgets v8対応）
# =================================================

print(
    '初期故障・偶発故障・摩耗故障の競合をシミュレーションし、ハザード関数とワイブルプロットを描画する\n' +
    '\n' +
    '手順:\n' +
    '1. 初期状態は「偶発故障」のみ。ワイブルの打点はほぼ直線上に乗る。\n'+
    '2. 初期故障を追加すると初期ハザードが上昇する。ワイブルの初期の打点は右に曲がる。近似直線の傾き(m)は穏やかになる。\n'+
    '3. 摩耗故障を追加するとハザードは右上がりになる。ワイブルの終期の打点は左に曲がる。近似直線の傾き(m)は急峻になる。\n'+
    '4. 初期故障＋摩耗故障ではハザードはバスタブ型になる。ワイブルは複雑なS字を描く。\n'
)

interact(
    simulate,
    w_early=widgets.FloatSlider(value=0.0, min=0.0, max=0.5, step=0.05, description="初期故障型のデータ点の割合率"),
    w_wear=widgets.FloatSlider(value=0.0, min=0.0, max=0.5, step=0.05, description="摩耗故障型のデータ点の割合"),
    n=widgets.IntSlider(value=300, min=10, max=1000, step=10, description="総データ点数")
)
