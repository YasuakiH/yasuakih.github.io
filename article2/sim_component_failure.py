#!python

# sim_component_failure.py
# Written in 2025 by yasuakih

'''
This is a demonstration of a printing press failure model implemented using SimPy, a process-based discrete-event simulation framework in Python.
これは、Pythonのプロセスベース離散事象シミュレーションフレームワークであるSimPyを使用して実装された印刷機の故障モデルのデモンストレーションである。

■参考
応力-強度モデル
https://reliability.readthedocs.io/en/latest/Stress-Strength%20interference.html

バスタブカーブの作成
https://reliability.readthedocs.io/en/stable/Creating%20and%20plotting%20distributions.html#example-4

■コマンドライン
usage:
python sim_component_failure.py

(例)
python sim_component_failure.py --designed_life 1000000 --wearout_rates 1.5 --maxt 60*24*30*12 --maxx 100
%run sim_component_failure.py --designed_life 1000000 --maxt 60*24*30*12 --beta 1.8 --eta 1000000 --wearout_rate 0.6 0.8 1.0 1.2 1.4 1.6 --ac L --pl L --iter 20
%run sim_component_failure.py --designed_life 1000000 --maxt 60*24*30*12 --beta 1.8 --eta 1000000 --wearout_rate 0.6 0.8 1.0 1.2 1.4 1.6 --ac L --pl M --iter 20
%run sim_component_failure.py --designed_life 1000000 --maxt 60*24*30*12 --beta 1.8 --eta 1000000 --wearout_rate 0.6 0.8 1.0 1.2 1.4 1.6 --ac L --pl H --iter 20


■補足
env.now の単位は [分] とした。これは規模感を与えるためのもので、厳密な意味の単位ではない。

'''
import sys
import simpy
import random
import statistics
import numpy as np
import scipy.stats as ss
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import argparse
import pandas as pd
import os
import shutil
import seaborn as sns
# import seaborn.objects as so
from addict import Dict
import logging
from datetime import datetime
import math
import pickle

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P
from reliability.Other_functions import stress_strength

pd.options.display.float_format = '{:.1f}'.format

this_file = 'sim_component_failure.py'  # このファイルのファイル名
import_file = ''  # ロジックとデータの分離。ファイルがなくても本スクリプトは動作する。
import_file_var = None

wait_times = None             # print_job 毎の印刷所要時間
printing_jobs_log = None      # print_job 毎の終了時刻と成否
replacement_parts_log = None  # 交換した部品 [交換理由, 停止時間, 部品情報]

def arg_parse():
    global args
    # global params
    parser = argparse.ArgumentParser()

    parser.add_argument('--step'          , action='store_true', default=False)
    parser.add_argument('--debug'         , action='store_true', default=False)
    parser.add_argument('--wearout_rates'  , type=str , nargs='+', default=['1.0'], help='予防保守の管理目標(係数)。部品ライフ設計値を1.0とした場合の管理目標(係数)を指定する。(デフォルト: 1.0)。(例: --wearout_rates 1.0, --wearout_rates 0.9 1.0 1.1)')
    parser.add_argument('--designed_life' , type=int  , default=1000000, help='部品ライフ設計値。算術平均やB(10)ライフなどで指定される (デフォルト: 1000000)。(例: --designed_life 1000000)')
    parser.add_argument('--beta'          , type=float, default=1.0, help='βは、部品ライフをワイブル分布で表した際の形状パラメータ。β＜1で初期故障型、β=1で偶発故障型、1<βで摩耗型故障を示す (デフォルト: 1.0)。(例: --beta 1.0)')
    parser.add_argument('--eta'           , type=int  , default=None, help='ηは、部品ライフをワイブル分布で表した際の尺度パラメータ。 (デフォルト: 部品ライフ設計値)。(例: --eta 1000000)')
    parser.add_argument('--check_interval', type=str  , default='60*24*10', help='保守計画における保守間隔 [単位:[分]] (デフォルト: 60*24*10 (10日間の意味))。(例: --check_interval 60*24*10)')
    parser.add_argument('--maxt'          , type=str  , default='60*24*30*12', help='シミュレーション期間 [単位:[分]] (デフォルト: 60*24*30*12 (1年間の意味))。(例: --maxt 60*24*30*12)')
    parser.add_argument('--maxx'          , type=int  , default=200, help='交換部品数の最大値。この指定に達した時点でシミュレーションを終了する (デフォルト: 200)。(例: --maxx 200)')
    parser.add_argument('--iter'          , type=int  , default=1, help='シミュレーション回数 (デフォルト: 1)。(例: --iter 10)')
    parser.add_argument('--seed'          , type=int, default=None, help='random.seed() 初期値。(デフォルト: None)。(例: --seed 42)')   # シミュレーションでは異なる乱数系列を用いて平均化する方が好ましい結果を得る
    parser.add_argument('--area_coverage', '--ac', type=str, default='M', help='area_coverage [L, M, H] (デフォルト: M)。(例: --area_coverage M)')
    parser.add_argument('--page_length'  , '--pl', type=str, default='M', help='page_length [L, M, H] (デフォルト: M)。(例: --page_length M)')

    args = parser.parse_args()
    if args.eta is None:
        args.eta = args.designed_life

    assert 1   <= args.designed_life      , f'設計値 --designed_life は 1 以上の int 値を指定する'
    assert 0.0 <  args.beta               , f'部品ライフの形状パラメータ --beta は 0.0 < beta の float 値を指定する'
    assert 1   <= args.eta                , f'部品ライフの尺度パラメータ --eta は 1 <= eta 以上の int 値を指定する'
    assert 1   <= eval(args.check_interval), f'保守間隔 --check_interval は 1 以上の値となる数値、あるいは計算式を指定する。[単位:分] (例: --check_interval 60*24*10)'
    assert 1   <= eval(args.maxt)          , f'シミュレーション期間 --maxt は 1 以上の値となる数値、あるいは計算式を指定する。[単位:分] (例: --maxt 60*24*30*12)'
    assert 1   <= args.maxx               , f'交換部品数の最大値 --maxx は 1 以上の値となる数値を指定する。(例: --maxx 200)'
    assert 1   <= args.iter               , f'シミュレーション回数 --iter は 1 以上の数値を指定する。(例: --iter 10)'
    assert (args.seed is None) or isinstance(args.seed, int), f'random.seed() 初期値 --seed は int 値を指定する。(例: --seed 42)'

    # print(f'args={args}')
    # sys.exit(1)

    # --wearout_rates v1 [v2, v3, ...] を解析して list を返す
    def parse_wearout_rates():
        '''
        --wearout_rates v1 [v2, v3, ...] を解析して list を返す
        例:
          --wearout_rates 0.5, 0.7, 0.9                   --> [0.5, 0.7, 0.9]
          --wearout_rates [0.5, 0.7, 0.9]                 --> [0.5, 0.7, 0.9]
          --wearout_rates [i/10 for i in range(5, 17, 1)] --> [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
          --wearout_rates [i/10 for i in range(5, 17, 2)] --> [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
          --wearout_rates 1                               --> [1.0]
        '''
        # print(f'args.wearout_rates={args.wearout_rates}')
        wearout_rate_joined = ' '.join(args.wearout_rates)
        try:
            wearout_rate_eval = eval(wearout_rate_joined)
        except SyntaxError:
            wearout_rate_eval = eval( wearout_rate_joined.replace(' ',',') )
        # print(f'wearout_rate_eval={wearout_rate_eval} ({type(wearout_rate_eval)})')
        wearout_rate_list = []
        if isinstance(wearout_rate_eval, float) or isinstance(wearout_rate_eval, int):
            wearout_rate_list = [float(wearout_rate_eval)]
        else:
            for x in wearout_rate_eval:
                # print(f'x={x} ({type(x)})')
                float_x = float(x)
                assert 0.0 < float_x <= 3.0, f'管理目標 --wearout_rates は 0.0 を超える、3.0 以下の float 値を 1つ以上指定する。異常値: {float_x}'
                wearout_rate_list.append(float_x)
        # print(f'wearout_rate_list={wearout_rate_list}')
        return wearout_rate_list
    # args.wearout_rates = parse_wearout_rates()

    # スクリプト実行中に args は固定したい。シミュレーションにパラメータを引き継ぐため dict 様の params を作成する
    params = Dict()  # Dict() パッケージはドット記法が可能
    params.step           = args.step
    params.debug          = args.debug
    params.wearout_rates  = parse_wearout_rates()
    params.designed_life  = args.designed_life
    params.beta           = args.beta
    params.eta            = args.designed_life
    params.check_interval = eval(args.check_interval)
    params.maxt           = eval(args.maxt)
    params.maxx           = args.maxx
    params.iter           = args.iter
    params.seed           = args.seed
    params.area_coverage  = args.area_coverage
    params.page_length    = args.page_length
    params.customer_hidden_parameters = '(AC=' + str(params.area_coverage) + ',PL=' + params.page_length + ')'

    assert params.area_coverage in ['L','M','H'], f'--area_coverage は L, M, H のいずれかを指定する (例: --area_coverage M)'
    assert params.page_length in ['L','M','H'], f'--page_length は L, M, H のいずれかを指定する (例: --page_length M)'

    # print(f'params={params}')
    # sys.exit(1)
    return params
# end-of def arg_parse

logger = None

def init_logging(logfile):
    '''ロギング'''
    global logger

    if logger is None:
        logger = logging.getLogger('sim')
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s [%(levelname).4s] [%(name)s]  %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        if (logger.hasHandlers()):
            logger.handlers.clear()

        logger.addHandler(fh)
        logger.addHandler(ch)
# end-of def init_logging

def my_gauss(mu, sigma, upper_limit, number_of_digits):
    '''離散的なガウス分布を生成する
    lower_limit 下限値
    upper_limit 上限値
    number_of_digits 小数点以下の桁数
      0: 印刷ジョブページ長で用いる (小数点以下は切り捨て)
      2: インクカバレッジと両面比で用いる (小数点以下 2桁を残して切り捨て)
    '''
    lower_limit = 0.0
    while True:
        v = random.gauss(mu, sigma)
        if number_of_digits == 0:
            v = float(int(v))
        elif number_of_digits == 2:
            v = float(int(v*100)/100)
        else:
            assert False, 'this must not be reached'
        if lower_limit < v <= upper_limit:
            break
    return v 

savedir_path = None
def distination_pathname(params, data_pathname='_debug', dt=False, filename='john.doe'):
    '''生成ファイルの格納先のパス名を返す。
    (例) distination_pathname(dt=True, filename='foo/bar/john.doe')
    (1) dt=True の場合、savedir_path に '_/debug/YYYYMMDD_HHMMSS' をストア (YYYYMMDD_HHMMSSは、初回実行日時。global変数に保存される)
        dt=False の場合、savedir_path に '_/debug' をストア
    (2) savedir_path にサブディレクトリを作成する
    (3) filename を '/' で分割し、head, tail に分ける。
        head があればさらにサブディレクトリを作成する。
    (4) dt=True の場合、本関数は tail を 'YYYYMMDD_HHMMSS-john.doe' とする。(YYYYMMDD_HHMMSSは、その時点での日時)
        dt=False の場合、本関数は tail を 'john.doe' とする
    (5) 呼び出し側は返された返されたパス名にファイルを作成する。
    '''

    # 格納先のサブディレクトリ作成
    def create_savedir():
        global savedir_path
        if dt:
            d = datetime.now()
            d_str = d.strftime('%Y%m%d_%H%M%S')
        else:
            d_str = ''
        savedir_path = os.path.join(data_pathname, d_str + params.customer_hidden_parameters + f'(iter={params.iter})')

        if not os.path.exists(savedir_path):
            os.makedirs(savedir_path)

    if savedir_path == None:
        create_savedir()

    head, tail = os.path.split(filename)
    if (head != '') and (not os.path.exists(os.path.join(savedir_path, head))):
        os.makedirs(os.path.join(savedir_path, head))

    if dt:
        d = datetime.now()
        return os.path.join(savedir_path, head, d.strftime('%Y%m%d_%H%M%S.%f') + '-' + tail)
    else:
        return os.path.join(savedir_path, head, tail)
# end-of def distination_pathname

figsize = (14, 12)  # (横, 縦)
font_normal = {'family' : 'IPAexGothic', 'weight' : 'bold', 'size' : 9}  # 日本語フォント
font_small  = {'family' : 'IPAexGothic', 'weight' : 'bold', 'size' : 7}  # 日本語フォント

def init_figure():
    '''新しい図を作成'''
    # matplotlib.use("Agg")
    fig = plt.figure(figsize=figsize)  # (横, 縦)
    matplotlib.rc('font', **font_normal)
    return fig
# end-of def init_figure

def print_t(env, s):
    if args.debug:
        logger.debug(f't={env.now:.2f}: {int(env.now/(60*24))}日 {s}')
        
class PrintJob():
    '''印刷ジョブ'''
    MAX_PAGE_LENGTH = 2000  # 印刷ジョブページ長の最大 (最小は1)
    MAX_SET_PER_JOB = 2000  # 印刷部数の最大 (最小は1)

    def generate_customer_print_job(self):
        '''顧客の未知パラメータに基づく印刷ジョブを作成
        印刷ジョブの属性:
        - 印刷用紙サイズ(重み付きランダム)
        - トータルエリアカバレッジ(用紙サイズにより分布は異なる)
        - 印刷ジョブ長(用紙サイズにより分布は異なる)
        - 両面/片面(μ=0.5, σ=0.3)
        '''


        # 印刷用紙サイズ (サイズ:割合の組、割合は合計1.0とする)
        #   値は次のシミュレーション結果に基づく: name=PM1 try=9  [TYP=Q4 AC=M PL=L]  [葉  0 A4  5 B4  3 A3 46 長 46] loop=8933  ink=100 枚数比=0.90 ce1=0.53 ce2=3.27 → OK (総OK:1)★
        customer_printed_matters = {'葉書': 0.0, 'A4': 0.05, 'B4':0.03, 'A3':0.46, '長尺':0.46}
        assert sum([customer_printed_matters[key] for key, value in customer_printed_matters.items()]) == 1.0, f'dict customer_printed_matters の value の合計を 1.0 とする。'
        paper_size = random.choices(
                list(customer_printed_matters.keys()),
                weights = list(customer_printed_matters.values())
        )[0]

        # トータルエリアカバレッジ
        # area_coverage_list = [0.10, 0.10]  # [平均(mu), 分散(sigma)]
        area_coverage_list = {
            '葉書': {'L': [0.10, 0.10], 'M': [0.20, 0.10], 'H': [0.20, 0.20]},
            'A4'  : {'L': [0.03, 0.05], 'M': [0.10, 0.10], 'H': [0.20, 0.20]},
            'B4'  : {'L': [0.03, 0.05], 'M': [0.04, 0.10], 'H': [0.05, 0.20]},
            'A3'  : {'L': [0.03, 0.05], 'M': [0.04, 0.10], 'H': [0.05, 0.20]},
            '長尺': {'L': [0.10, 0.10], 'M': [0.30, 0.10], 'H': [0.50, 0.20]},
        }
        mu, sigma = area_coverage_list[paper_size][self.area_coverage]
        area_coverate = my_gauss(mu, sigma, upper_limit=2000, number_of_digits=2)
        # print_t(self.env, f'  area_coverate={area_coverate}')

        # 印刷ジョブ長
        # page_length_list = [300, 300]  # [平均(mu), 分散(sigma)]
        page_length_list = {
            '葉書': {'L': [0,  50], 'M': [  0, 100], 'H': [  0, 200]},
            'A4'  : {'L': [0, 300], 'M': [300, 300], 'H': [500, 600]},
            'B4'  : {'L': [0, 300], 'M': [300, 300], 'H': [500, 600]},
            'A3'  : {'L': [0, 300], 'M': [300, 300], 'H': [500, 600]},
            '長尺': {'L': [0,   5], 'M': [  0,  10], 'H': [  0,  30]},
        }
        page_length_mu, page_length_sigma = page_length_list[paper_size][self.page_length]
        page_length = int(my_gauss(page_length_mu, page_length_sigma, self.MAX_PAGE_LENGTH, 0))
        # print_t(self.env, f'  page_length={page_length}')

        # 両面/片面
        duplex_rate_list = [0.5, 0.3]  # [平均(mu), 分散(sigma)]
        duplex_rate_mu, duplex_rate_sigma = duplex_rate_list
        self.duplex_rate = my_gauss(duplex_rate_mu, duplex_rate_sigma, 1.0, 2)
        if 0.5 <= self.duplex_rate:
            duplex_or_simplex = 'duplex'
        else:
            duplex_or_simplex = 'simplex'

        # print_t(self.env, f'  ' + str((area_coverate, paper_size, page_length, duplex_or_simplex)))
        return (area_coverate, paper_size, page_length, duplex_or_simplex)

    def __init__(self, env, id, area_coverage=0.1, paper_size='A4', page_length=1, duplex_or_simplex='simplex'):
        '''印刷ジョブ作成'''
        self.id             = id
        self.env            = env

        # 未知パラメータ
        self.area_coverage  = env.params.area_coverage  # ['L','M','H'] のいずれか
        self.page_length    = env.params.page_length    # ['L','M','H'] のいずれか

        customer_print_job = self.generate_customer_print_job()  # 顧客の未知パラメータに基づく印刷ジョブを作成

        (   self.area_coverage,
            self.paper_size,
            self.page_length,
            self.duplex_or_simplex
        ) = customer_print_job
        print_t(env, f'印刷ジョブを作成: {self.__str__()}')

    def __str__(self):
        return f'[#{self.id} AC{self.area_coverage} PS={self.paper_size} LEN={self.page_length} {self.duplex_or_simplex}]'
# end-of class PrintJob

class ReplacementPart():
    '''交換部品 - 交換部品の生成, 部品ライフ進行(摩耗), 故障確率の算出

    単位:
      部品ライフを [A4短辺ページ] で表現する。用紙サイズの異なりを補正するため、[A4短辺ページ] は「A4短辺」を1とした無次元の量である。

    例:
      用紙サイズ=A3、ジョブ長=3の場合、部品にかかる負荷 ΔT = 用紙長比2.30 × ジョブ長3 = 6.90 [A4短辺ページ] となる。
    '''

    # 用紙長比 (各用紙の1枚あたりの長さ。A4短辺を1とする。)
    paper_length_ratio = {
        '葉書' : 148/210,    # 0.704 | 葉書長辺/A4短辺 | タテ置き
        'A4'   : 210/210,    # 1.00  |   A4短辺/A4短辺 | ヨコ置き
        'B4'   : 364/210,    # 1.73  |   B4長辺/A4短辺 | タテ置き
        'A3'   : 483/210,    # 2.30  |   A3長辺/A4短辺 | タテ置き
        '長尺' : 1200/210,   # 5.71  | 長尺長辺/A4短辺 | タテ置き
    }

    parts_life = None  # 偽のライフ実績 (list)
 
    def set_part_life_distribution(self):
        '''部品ライフ分布を生成(ワイブル分布)
        印刷機全体の母集団における部品ライフを規定する。部品強度 F を与える。
        本来は保守サービスを介して収集した部品ライフに基づいて設定するところだが、架空の印刷機のものとしてワイブル分布を仮定した。
        部品ライフは無次元してA4短辺を1とした。
        '''
        # (1) 正規分布
        # return int(random.gauss(1000000, 100000))  # 正規分布は負の部品ライフを生成するため適当でない
        # 
        # (2) ワイブル分布 (部品ライフ実績から推定する場合を想定。このライブラリは打ち切りを含むデータに対応している。)
        # if self.parts_life is None:
        #     def get_dummy_failures():
        #         '''(仮の) 部品ライフ実績を生成する (ここではワイブル分布からサンプリングした(20件))'''
        #         wd = Weibull_Distribution(alpha=self.env.params.eta, beta=self.env.params.beta)
        #         # wd.plot()  # 全関数 (PDF, CDF, SF, HF, CHF) の表示
        #         parts_life = wd.random_samples(20)
        #         if self.env:
        #             print_t(self.env, f'eta={self.env.params.eta} beta={self.env.params.beta} parts_life={parts_life}')
        #         return parts_life
        #     parts_life = get_dummy_failures()  # (仮の) 部品ライフ実績を生成する
        # 
        #     # ライフ実績をもとにワイブル分布を生成 (打ち切りはなし(right_censored=None)と仮定した)
        #     self.wd = Fit_Weibull_2P(failures=parts_life, right_censored=None, show_probability_plot=False, print_results=False).distribution
        #     self.wd.plot()  # 全関数 (PDF, CDF, SF, HF, CHF) の表示
        # 
        # (3) ワイブル分布 (alpha, beta が既知の場合)
        if self.parts_life is None:
            # logger.debug(f'alpha={self.env.params.eta} ({type(self.env.params.eta)})')
            # logger.debug(f'beta ={self.env.params.beta} ({type(self.env.params.beta)})')
            # sys.exit()
            self.wd = Weibull_Distribution(alpha=self.env.params.eta, beta=self.env.params.beta)
            if self.env.params.step:
                self.wd.plot()  # 全関数 (PDF, CDF, SF, HF, CHF) の表示  (交換時のステップ実行をしたいとき、--step 付きで実行すると交換の都度グラフが表示される)

    def __init__(self, env, replacement_reason):
        '''交換部品の生成'''
        self.env             = env
        self.set_part_life_distribution()  # 部品ライフ分布を生成(ワイブル分布)

        assert isinstance(self.env.params.wearout_rate, float), f'self.env.params.wearout_rate must be an float: type(self.env.params.wearout_rate)={type(self.env.params.wearout_rate)}'

        self.life_limit      = int(        # 交換時の管理目標 [A4短辺ページ]
            self.env.params.designed_life *         # 部品ライフ設計値 [A4短辺ページ]
            self.env.params.wearout_rate            # 部品ライフ設計値を1.0とした場合の管理目標(係数)
        )
        self.cum_page_length_before = None  # 印刷ジョブ出力前の 累積印刷ページ数 [A4短辺ページ]
        self.cum_page_length_after  = 0     # 印刷ジョブ出力後の 累積印刷ページ数 [A4短辺ページ]

        self.survival_prob_before = None    # 印刷ジョブ出力前の生存確率

        assert replacement_reason in ['予防保守', '障害修理']
        self.replacement_reason = replacement_reason  # 交換理由

        self.replaced_time = int(env.now)      # 交換部品を生成した日時

        if env:
            print_t(self.env, f'      交換部品を設置: {self.__str__()}')

    def info(self):
        return {
            '交換日時'        : self.replaced_time,
            '部品ID'         : self.replaced_time,   # 部品固有のID (やっつけで交換日時を利用)
            '計画部品ライフ'   : self.life_limit,
            '累積印刷ページ数(ジョブ出力前)' : self.cum_page_length_before,
            '累積印刷ページ数(ジョブ出力後)' : self.cum_page_length_after,
            '交換理由'        : self.replacement_reason,
        }

    def wear(self, print_job):
        '''部品ライフ進行 (摩耗)
        累積印刷ページに「ページ長」を加算し、部品ライフを進行させる。
        ライフ進行の推定で利用可能な「未知パラメータ」:
          - self.area_coverage        トータルエリアカバレッジ
          - self.paper_size           用紙サイズ
          - self.page_length_before   印刷ページ長 before [A4短辺ページ]
          - self.page_length_after    印刷ページ長 after  [A4短辺ページ]
          - self.duplex_or_simplex    両面片面
        '''
        CLEARNING_PAGES_CYCLE_PER_JOB  = 100  # 印刷ジョブ中のクリーニング間隔 [A4短辺ページ/回]。ジョブ長がこれより長い場合に定期的に挿入する。
        CLEARNING_PAGES_LENGTH_PER_JOB =   1  # 印刷ジョブ終了後のクリーニングによる経年 [A4短辺ページ]。ジョブ長が短くなるにつれて重みは増す。

        # 現在の部品ライフ
        self.cum_page_length_before = self.cum_page_length_after  # 印刷ジョブ出力前の累積印刷ページ数を保存 (メソッド failure(self) で使用する)

        # 経年の計算
        # (1)印刷ジョブによる経年: [A4短辺ページ]
        pages_per_print_job = print_job.page_length * self.paper_length_ratio[print_job.paper_size]  # 「印刷ジョブページ長」×「用紙長比」
                                                                                                     # なお、両面/片面の別は考慮しない (片面ずつ印刷すると仮定)
        # (2)クリーニングによる経年: [A4短辺ページ]
        clearning_pages_per_print_job = (
            math.floor(pages_per_print_job / CLEARNING_PAGES_CYCLE_PER_JOB) +  # ジョブ中に定期的に挿入されるクリーニング (小数点以下は切り捨て)
            CLEARNING_PAGES_LENGTH_PER_JOB  # ジョブ終了後に付加されるクリーニング
        )

        # (3)印刷ジョブ後の部品ライフの計算
        self.cum_page_length_after += math.floor(  # 印刷ジョブ出力後の累積印刷ページ数に加算 (小数点以下は切り上げ)
            pages_per_print_job +
            clearning_pages_per_print_job
        )

        print_t(self.env, f'      累積印刷ページ数(ジョブ出力前): cum_page_length_before={self.cum_page_length_before} → 同(ジョブ出力後)cum_page_length_after={self.cum_page_length_after}')

    def failure(self):
        '''故障確率の算出と生存-故障判断。部品が生存するか(False)、故障するか(True)を判断して返す。
        部品強度 R に対応する生存関数(SF)を元に、印刷ジョブ出力前まで生き残った部品がさらに印刷ジョブの出力後まで生き残る確率 (条件付き生存率CS) を算出した。
        故障か故障でないか確率的に決定するために一様乱数を使用した。
        '''
        print_t(self.env, f'      self.cum_page_length_before = {self.cum_page_length_before}')
        print_t(self.env, f'      self.cum_page_length_after  = {self.cum_page_length_after} (delta={ "----" if self.cum_page_length_before is None else self.cum_page_length_after - self.cum_page_length_before })')

        # 印刷ジョブ出力前の生存確率を生存関数(SF)から得る
        if self.cum_page_length_before is None:
            self.survival_prob_before = 1.0
        else:
            if self.survival_prob_before is None:
                self.survival_prob_before = self.wd.SF( self.cum_page_length_before )
            else:
                pass  # 何もしない (前回の印刷ジョブ出力の計算結果を利用する)

        # 印刷ジョブ出力後の生存確率を生存関数(SF)から得る
        survival_prob_after  = self.wd.SF( self.cum_page_length_after )

        print_t(self.env, f'      survival_prob_before        = {self.survival_prob_before}')
        print_t(self.env, f'      survival_prob_after         = {survival_prob_after} (delta={survival_prob_after - self.survival_prob_before})')

        conditional_survival = survival_prob_after / self.survival_prob_before   # 生存確率 (印刷ジョブ開始後における条件付き生存確率)
        print_t(self.env, f'      conditional_survival        = {conditional_survival} ')

        # 故障か故障でないか確率的に決定する (一様乱数を利用)
        uniform_random_number = random.random()                 # 一様乱数を生成
        failure = conditional_survival < uniform_random_number  # 故障か故障でないか確率的な決定

        print_t(self.env, f'      累積印刷ページ数(ジョブ出力後)={self.cum_page_length_after} (交換時の管理目標 {self.life_limit} に対する比率: {(self.cum_page_length_after/self.life_limit*100):.2f}%) 条件付き生存確率={conditional_survival:0.5f} 一様乱数={uniform_random_number:0.5f} failure={failure} → { "故障★" if failure else "生存" }')

        # (次回の印刷ジョブ出力に備えて) 「印刷ジョブ出力後」の生存確率を保存 (処理時間を削減するため)
        self.survival_prob_before = survival_prob_after

        return failure

    def __str__(self):
      return f'[交換日時={self.replaced_time} 部品ID={self.replaced_time} 計画部品ライフ={self.life_limit} 累積印刷ページ数(ジョブ出力後)={self.cum_page_length_after} 交換理由={self.replacement_reason}]'
# end-of class ReplacementPart

# 印刷機の保守計画
class MaintenanceWork():
    '''保守作業'''
    def __init__(self, env, printer, num_engineers=1):
        self.env = env
        self.printer = printer
        print_t(self.env, f'保守作業init')
        self.customer_engineer = simpy.Resource(env, capacity=num_engineers) # 環境にリソース追加(保守エンジニア)

    def preventive_maintenance_setup_process(self, check_interval):
        '''印刷機の予防保守のスケジュールと実施プロセス
        予防保守の作業を記述する。予防保守の実施間隔 (check_interval) は、保守サービスの管理目標値として規定される (デフォルト: 10日間)。予防保守の作業内容は、部品ライフが計画部品ライフを超えていたら部品を交換し、次回の予防保守をスケジュールする。なお、交換の際はリソース (エンジニアと印刷機ユニット) の確保を要するとした。
        '''
        def local_print_t(s):
            print_t(self.env, s)
            pass
        # end-of def local_print_t
        local_print_t(f'■(予防保守)計画: BEGIN : {self.printer.replacement_part}')

        next_preventive_maintenance_time = self.env.now + check_interval  # 次回の予防保守の予定日
        local_print_t(f'■(予防保守)待機: 次回check t = {next_preventive_maintenance_time}')

        # 次回の予防保守の日時が来るまで待機
        yield self.env.timeout(check_interval)   # 次回の予防保守まで待機 (時間: check_interval)

        # 現在部品ライフが計画部品ライフを超えているかいないか判断
        page_length_diff = (
            self.printer.replacement_part.cum_page_length_after - 
            self.printer.replacement_part.life_limit
        )
        local_print_t(f'■(予防保守)再開: check {self.printer.replacement_part} page_length_diff = {page_length_diff:.1f}')
        # 計画部品ライフを超過したら部品を交換
        if 0 <= page_length_diff: 
            local_print_t(f'■(予防保守)交換: 計画部品ライフを超えたので部品を交換する')

            # (予防保守)部品を交換するエンジニアを確保
            with self.printer.customer_engineers.request() as request:
                local_print_t(f'■(予防保守)     エンジニアを確保 request開始')
                yield request  # raise a event
                local_print_t(f'■(予防保守)     エンジニアを確保 request終了')

                # (予防保守)印刷機ユニットを確保
                with self.printer.printing_units.request() as request:
                    local_print_t(f'■(予防保守)      印刷機ユニットを確保 request開始')
                    yield request  # raise a event
                    local_print_t(f'■(予防保守)      印刷機ユニットを確保 request終了')

                    # 予防保守実施
                    local_print_t(f'■(予防保守)      エンジニア作業開始')
                    yield self.env.process(self.printer.preventive_maintenance_process())  # 予防保守実行プロセス
                    local_print_t(f'■(予防保守)      エンジニア作業終了')
                # end-of with self.printer.printing_units.request

                local_print_t(f'■(予防保守)      エンジニア開放')
            # end-of with self.printer.customer_engineers.request
            local_print_t(f'■(予防保守)交換: {self.printer.replacement_part}')
        # end-of if 0 <= page_length_diff

        # 次回の予防保守 (今回、交換しても交換しなくても、次回の計画を要する)
        self.env.process(self.preventive_maintenance_setup_process(check_interval))  # 印刷機の予防保守のスケジュールと実施プロセス
        
        local_print_t('■(予防保守)完了: END')
# end-of class MaintenanceWork

# 印刷機ユニット
class PrintingMachine(object):
    '''印刷機ユニット'''

    PRINTING_SPEED = 30   # 印刷速度 [A4短辺ページ/分]

    def __init__(self, env, id, num_printing_units=1, num_engineers=1):
        self.env = env
        self.id  = id
        self.printing_units = simpy.Resource(env, capacity=num_printing_units) # 環境にリソース追加(印刷機ユニット作成)
        self.customer_engineers = simpy.Resource(env, capacity=num_engineers)  # 環境にリソース追加(保守エンジニア)

    def terminate_simulation(self):
        '''交換部品が一定数に達したので (シミュレーション期間が残っていても) シミュレーションを終了する'''
        # print_t(self.env, f'len(replacement_parts_log)={len(replacement_parts_log)}\tself.env.params.maxx={self.env.params.maxx}')
        if self.env.params.maxx == len(replacement_parts_log):
            print_t(self.env, f'交換部品数 ({len(replacement_parts_log)}) が一定数 (self.env.params.maxx={self.env.params.maxx}) に達したのでシミュレーションを終了する')
            end_event.succeed()

    def preventive_maintenance_process(self):
        '''予防保守実行プロセス
        エンジニアによる部品の交換を記述した。計画内の作業であるため印刷機を止める作業時間を短くした (30分)。
        '''
        print_t(self.env, f'    予防保守: BEGIN')
        # インストールされた交換部品を記録
        try:
            before_replacement_time = self.env.now                   # 交換前日時
            before_replacement_part = self.replacement_part.info()   # 交換前部品
        except AttributeError:  # 印刷機ユニット作成後、初回の部品のインストール時にこの例外が起こる (self.replacement_part が存在しないため)
            before_replacement_part = None
            pass

        # 交換部品の生成
        self.replacement_part = ReplacementPart(self.env, replacement_reason='予防保守')
        # 作業時間を加算
        yield self.env.timeout(random.randint(30, 30))  # raise a event  # 作業時間待機 (時間: 30分)

        # 停止時間(計画内)の計算
        down_time = int(self.env.now - before_replacement_time)

        if before_replacement_part is None:
            # 初回の部品のインストールである場合は、(交換ではないため) 交換部品の情報を記録しない
            before_replacement_part = self.replacement_part.info()
        else:
            # 初回の部品のインストールでない場合、すなわち通常の予防保守交換である場合は、交換部品の情報を記録する

            # 停止時間(計画内)の記録
            replacement_parts_log.append({'T': self.env.now, '理由': '予防保守', '停止時間': down_time, '交換前部品': before_replacement_part})
        # end-of if

        # 交換部品が一定数に達したので (シミュレーション期間が残っていても) シミュレーションを終了する
        self.terminate_simulation()
        print_t(self.env, f'    予防保守: END')
        # sys.exit(1)

    def corrective_maintenance_process(self):
        '''障害修理実行プロセス
        予防保守と同様に、エンジニアによる部品の交換であるが、計画外の作業であるため印刷機を止める作業時間を長くした (60-90分)。
        '''
        print_t(self.env, f'    障害修理: BEGIN')
        # インストールされた交換部品を記録
        try:
            before_replacement_time = self.env.now                   # 交換前日時
            before_replacement_part = self.replacement_part.info()   # 交換前部品
        except AttributeError:
            pass
        print_t(self.env, f'    障害修理: before_replacement_time = {before_replacement_time} (交換前日時)')
        print_t(self.env, f'    障害修理: before_replacement_part = {before_replacement_part} (交換前部品)')

        # 交換部品の生成
        self.replacement_part = ReplacementPart(self.env, replacement_reason='障害修理')
        # 作業時間を加算
        yield self.env.timeout(random.randint(60, 90))  # raise a event  # 作業時間待機 (時間: 60-90分)

        # 停止時間(計画外ダウンタイム)の計算
        down_time = int(self.env.now - before_replacement_time)

        # 停止時間(計画外ダウンタイム)の記録
        replacement_parts_log.append({'T': self.env.now, '理由': '障害修理', '停止時間': down_time, '交換前部品': before_replacement_part})
        print_t(self.env, f'    障害修理: replacement_parts_log = {replacement_parts_log}');

        # 交換部品が一定数に達したので (シミュレーション期間が残っていても) シミュレーションを終了する
        self.terminate_simulation()
        print_t(self.env, f'    障害修理: END')
        # sys.exit(1)

    def printout_process(self, print_job):
        '''印刷実行プロセス(含む部品ライフ進行(摩耗))
        印刷ジョブを出力を記述する。印刷の所要時間は、印刷ジョブ長/印刷速度 とした。その後、部品ライフを進行させた。
        '''
        # 印刷ジョブ出力 (出力時間の待機)
        print_t(self.env, f'    印刷ジョブの出力: BEGIN {print_job}')
        yield self.env.timeout(
            print_job.page_length / self.PRINTING_SPEED
        )  # raise a event  # 印刷時間待機 (時間: 印刷ジョブ長/印刷速度)

        # 部品ライフ進行 (摩耗)
        self.replacement_part.wear(print_job)
        print_t(self.env, f'    印刷ジョブの出力: END   {print_job}')

    def __str__(self):
        return f'{self.id}'
# end-of class PrintingMachine

def printing_printjob_process(env, print_job, printer):
    '''印刷ジョブ出力プロセス'''

    begin_time = env.now    # print_job の到着日時
    print_t(print_job.env, f'  印刷ジョブ到着: {print_job}')

    # 印刷ジョブ出力のため、印刷機ユニットを確保
    with printer.printing_units.request() as request:
        print_t(print_job.env, f'    印刷機ユニットを確保 request開始')
        yield request  # raise a event
        print_t(print_job.env, f'    印刷機ユニットを確保 request終了')

        # 故障確率の算出と生存-故障判断
        if printer.replacement_part.failure():
            succeeds = False
            print_t(print_job.env, f' ★故障')
            # 故障時、修理するエンジニアを確保
            with printer.customer_engineers.request() as request:
                print_t(print_job.env, f'    エンジニアを確保 request開始')
                yield request  # raise a event
                print_t(print_job.env, f'    エンジニアを確保 request終了')
                # 障害修理実行プロセス
                yield env.process(printer.corrective_maintenance_process())  # raise a event  # 障害修理実行プロセス
                print_t(print_job.env, f'    エンジニア開放')
            print_t(print_job.env, f' ★回復  ')
        else:
            succeeds = True
        # end-of if printer.replacement_part.failure()

        # 印刷ジョブを出力
        yield env.process(printer.printout_process(print_job))  # raise a event  # 印刷実行プロセス(含む部品ライフ進行(摩耗))

        # print_job の印刷を完了
        wait_times.append(env.now - begin_time)  # print_job 毎の印刷所要時間を記録
        print_t(print_job.env, f'    印刷機ユニットを開放')
    # end-of with printer.printing_units.request() as request
    # 印刷機を開放する

    print_t(print_job.env, f'  印刷ジョブ終了: {print_job}    succeeds = {succeeds} {"成功" if succeeds else "故障"}')
    printing_jobs_log.append([env.now, succeeds])  # print_job 毎の終了時刻と成否を記録
# end-of def printing_printjob_process

def printingmachine_simulator_process(env, num_printing_units, num_engineers):
    '''印刷シミュレーションプロセス
    シミュレーション環境を構築し、さまざまな初期化をした後、シミュレーション中の印刷ジョブを生成する。シミュレーションは内部時計が上限を超過するか、交換部品数が所定数に達したら終了する。
    '''

    # 印刷機ユニット作成
    print_t(env, f'印刷機ユニット作成: BEGIN')
    printing_machine_id = 'PM1'
    printer = PrintingMachine(env, printing_machine_id)
    # 印刷機ユニットを確保
    with printer.printing_units.request() as request:
        yield request
        # 部品の初回インストール。
        print_t(env, f'印刷機ユニット作成: 部品の初回インストール: BEGIN')
        env.process(printer.preventive_maintenance_process())  # 予防保守実行プロセス (平行動作)
        print_t(env, f'印刷機ユニット作成: 部品の初回インストール: END')
    print_t(env, f'印刷機ユニット作成: END')

    # 印刷機の保守計画を作成 (実施間隔: check_interval)
    print_t(env, f'印刷機の保守計画を作成: BEGIN')
    maintenance_work = MaintenanceWork(env, printer)
    with maintenance_work.customer_engineer.request() as request:
        yield request

        check_interval = env.params.check_interval   # 実施間隔 [日]  (例: 60*24*10 )
        print_t(env, f'印刷機の保守計画: check_interval = {check_interval}')

        env.process(
            maintenance_work.preventive_maintenance_setup_process(check_interval = check_interval)
        )  # 印刷機の予防保守のスケジュールと実施プロセス (平行動作)
    print_t(env, f'印刷機の保守計画を作成: END')
    # sys.exit()

    # シミュレーション開始時点で存在する印刷ジョブ生成
    print_t(env, f'シミュレーション開始時点で存在する印刷ジョブ生成')
    print_job_id = 0
    initial_jobs = 1
    for print_job_id in range(initial_jobs):
        print_job = PrintJob(env, print_job_id)  # 印刷ジョブ生成
        env.process(printing_printjob_process(env, print_job, printer))  # 印刷ジョブ生成プロセス (平行動作)

    # シミュレーション期間中に受注する印刷ジョブ生成
    print_t(env, f'シミュレーション期間中に受注する印刷ジョブ生成')
    while True:   # 受注待ち

        if env.params.maxt <= env.now:
            print_t(env, f'シミュレーション日時 ({env.now}) が一定数 (env.params.maxt={env.params.maxt}) に達したのでシミュレーションを終了する')
            end_event.succeed()

        wait_min = 30  # [分]
        yield env.timeout(wait_min)  # raise a event  # 受注待ち待機 (時間: 30分)

        print_job_id += 1
        print_job = PrintJob(env, print_job_id)  # 印刷ジョブ生成
        env.process(printing_printjob_process(env, print_job, printer))  # 印刷ジョブ生成プロセス (平行動作)
# end-of def printingmachine_simulator_process

# 次のいずれかでシミュレーションを止める
# (1) シミュレーション日時 ({env.now}) が一定数 (env.params.maxt={env.params.maxt}) に達する
# (2) 交換部品数 ({len(replacement_parts_log)}) が一定数 (env.params.maxx={env.params.maxx}) に達する
end_event = None  # シミュレーションを終了させるイベント

# シミュレーション実行条件を表示
def simulation_parameters_str(params):
    result = (
        # f'args={args}' + '\n'\
        # f'管理目標の係数: {params.wearout_rate}' + ' ' + 
        f'部品ライフ設計値: {int(params.designed_life/1000)}k [A4短辺ページ]' + ' ' + 
        f'(β={params.beta}' + ' ' + 
        f',η={int(params.eta/1000)}k)' + ' ' + 
        # f'{params.check_interval}' + ' ' + 
        # f'{params.maxt}' + ' ' + 
        # f'{params.maxx}' + ' ' + 
        # f'iter={params.iter}'
        ''
    )
    return result
# end-of def simulation_parameters_str

def my_stress_strength(stress, strength, show_plot=True, print_results=True, warn=True, **kwargs):
    '''
   【重要】 stress_strength() 関数のグラフ表示に難があり、表示が欠落する不具合を起こすことがあった。
    この不具合を防ぐために、reliability の Other_functions.py を、差分ファイル「Other_functions.py.diff」のように修正したが、
    それをしない場合でも動作するよう、起こりうる TypeError を捕捉した。
    正しく動作させるためには Other_functions.py に差分を適用すること (手作業で容易)。
    '''
    try:
        # kwargs パラメータを拡張した reliability.Other_functions.stress_strength 関数を呼び出す。
        probability_of_failure = stress_strength(
            stress        = stress,
            strength      = strength,
            show_plot     = show_plot,
            print_results = print_results,
            warn          = warn,
                          **kwargs,    # 拡張したパラメータ
        )
        logger.debug('kwargs パラメータを拡張した reliability.Other_functions.stress_strength 関数を使用した')
        return probability_of_failure
    except TypeError:
        # 失敗した場合、オリジナルの reliability.Other_functions.stress_strength 関数を使用
        probability_of_failure = stress_strength(
            stress        = stress,
            strength      = strength,
            show_plot     = show_plot,
            print_results = print_results,
            warn          = warn,
        )
        logger.debug('オリジナルの reliability.Other_functions.stress_strength 関数を使用した')
        return probability_of_failure
# end-of def my_stress_strength

# 応力-強度干渉グラフ作成
def show_stress_strength_chart(params, result_all_df):
    '''応力-強度干渉グラフ作成'''

    if len(result_all_df) == 0:
        logger.debug(f'シミュレーション終了: サンプルがなく、平均処理時間は算出できず')
        return

    # 応力-強度干渉グラフ作成
    # -----------------------

    # 分析用データ作成
    def create_analysis_data(each_wearout_rate, result_all_df):
        # 分析用データ作成
        mask = result_all_df['管理目標(係数)'] == each_wearout_rate
        parts_life_data = {
            'failures'      : result_all_df.loc[mask & (result_all_df['理由'] == '障害修理'), '累積印刷ページ数[A4短辺ページ]'].tolist(), # 障害修理による交換部品ライフ
            'right_censored': result_all_df.loc[mask & (result_all_df['理由'] == '予防保守'), '累積印刷ページ数[A4短辺ページ]'].tolist(), # 予防保守による交換部品ライフ
        }
        return parts_life_data

    # 部品強度 (青) - parts_strength
    def plot_strength_chart(each_wearout_rate, xvals):
        # 部品強度 (青) - parts_strength
        # --------------------------------
        parts_strength_alpha = params.eta
        parts_strength_beta  = params.beta

        b10 = ss.weibull_min.ppf(0.10, parts_strength_beta, scale=parts_strength_alpha, loc=0.0)
        b50 = ss.weibull_min.ppf(0.50, parts_strength_beta, scale=parts_strength_alpha, loc=0.0)

        # '{:.0f}'.format() はfloat値をエンジニアリング表記(例:1E3)で表示されることを回避する
        label = f'強度\n [ワイブル分布] (α=' + '{:.0f}k'.format(parts_strength_alpha/1000) + f' β={parts_strength_beta:.1f})'

        parts_strength_dist = Weibull_Distribution(alpha=parts_strength_alpha, beta=parts_strength_beta)
        parts_strength = parts_strength_dist.PDF(xvals=xvals, label=label, color='b')

        children = plt.gca().get_children()
        designed_life_line = plt.axvline(params.designed_life                , c='gray', linestyle='-')   # 部品ライフ設計値を示す縦線を表示 (実線)

        designed_life_b10_line = plt.axvline(b10             , c='gray', linestyle='dotted')   # B(10)ライフを示す縦線を表示 (実線)
        designed_life_b50_line = plt.axvline(b50             , c='gray', linestyle='dotted')   # B(50)ライフを示す縦線を表示 (実線)

        ylims = plt.ylim(auto=None)
        plt.text(b10, ylims[1] * 0.1,' B(10)\n' + ' {:.0f}k'.format(b10/1000))
        plt.text(b50, ylims[1] * 0.1,' B(50)\n' + ' {:.0f}k'.format(b50/1000))
        plt.text(params.eta, ylims[1] * 0.1, ' B(63.3)\n' + ' {:.0f}k'.format(parts_strength_alpha/1000))
        # plt.text(params.eta, ylims[1] * 0.8, ' 設計値 (α={:.0f}k [A4短辺ページ])'.format(params.designed_life/1000))

        title_str =(
            '(予防保守の管理目標: 設計値 × 係数 = {:.0f}k'.format(params.designed_life/1000) +
            f' × {each_wearout_rate} = ' +
            '{:.0f}k'.format((params.designed_life * each_wearout_rate)/1000) +
            f') {params.customer_hidden_parameters}'
        )
        plt.title(f'応力-強度モデル {title_str}')
        plt.xlabel('')
        plt.ylabel('強度の確率密度 (PDF)')
        plt.xscale('log')
        plt.legend(
            [   children[0],
                # designed_life_line
            ],
            [   label,
                # '設計値 (α={:.0f}k [A4短辺ページ])'.format(params.designed_life/1000)
            ])
        plt.xlim(params.designed_life * 0.01, params.designed_life * 2.1)

        return parts_strength_dist

    # 応力 (赤) - failures_and_survivers
    def plot_stress_chart(each_wearout_rate, xvals, parts_life_data, show_labels=True):
        '''「応力-強度モデル」でいう応力側の分布を故障データから作成する。'''

        # 応力は顧客の印刷機の使われ方に対応し、交換理由 (障害修理、予防保守) を問わず、交換された部品全体のライフ分布を代表するものである。
        # 本スタディでは応力の確率分布を対数正規分布とした。
        # 理由: 対数正規分布は保守可能なシステムの修復時間をモデル化するためによく使用される。
        #      (参考) O'Connor, Patrick; Kleyner, Andre (2011). Practical Reliability Engineering. John Wiley & Sons. p.35. ISBN 978-0-470-97982-2
        # なお、フィッティング誤差ではワイブル > ガンマ > 対数正規の順に優れていたが、ワイブル関数は故障モデルに使用したためよく適合したと考えられる。ワイブル関数はガンマ関数を一般化したもの。
        parts_exchange_dist = Fit_Lognormal_2P(
            failures = parts_life_data['failures'],             # 交換データ
            right_censored = parts_life_data['right_censored'], # 生存データ
            show_probability_plot = False,
            print_results=False,
        ).distribution
        # parts_exchange_dist.plot()

        label = (
            '応力 (予防保守→生存, 障害修理→故障)\n [対数正規分布] (' +
              f'μ={parts_exchange_dist.mu:.2f} ' +
              f'σ={parts_exchange_dist.sigma:.2f}' +
            ')'
        )
        failures_and_survivers = parts_exchange_dist.PDF(xvals=xvals, show_plot=True, label=label, color='r')  # 応力カーブ (赤色, 実線)

        # 障害修理と予防保守による交換部品ライフの算術平均。グラフ上に参考として表示するために計算。
        part_life_list = (
            parts_life_data['failures'] +       # 交換
            parts_life_data['right_censored']   # 生存
        )
        part_life_mean = np.mean(part_life_list)  # 障害修理と予防保守による交換部品ライフの算術平均。
        part_life_num  = len(part_life_list)      # データポイント数

        ylims = plt.ylim(auto=None)

        lognormal_mu_line = plt.axvline(part_life_mean, c='green', linestyle='dotted')  # 応力の平均値 (交換部品ライフの算術平均) (緑色, 点線)
        plt.text(part_life_mean, ylims[1] * 0.85, ' 交換された部品ライフの算術平均\n  ({:.0f}k, N={})'.format(round(part_life_mean/1000,0), part_life_num))

        children = plt.gca().get_children()
        pm_target_line    = plt.axvline(params.designed_life * each_wearout_rate, c='green', linestyle='-', linewidth=2)  # 管理目標 (緑色 実線)
        plt.text(params.designed_life * each_wearout_rate, ylims[1] * 0.65, ' 予防保守の管理目標\n  ({:.0f}k)'.format(round(params.designed_life * each_wearout_rate/1000,0)))

        plt.title(f'')
        plt.xlabel('')
        if show_labels:
            plt.ylabel(f'応力の確率密度 (PDF)')
        else:
            plt.ylabel(f'')
        plt.xscale('log')

        # 凡例は常に表示する (対数正規分布のパラメータを表示したい)
        plt.legend(
            [   children[0],
                # lognormal_mu_line,
                # pm_target_line,
            ],
            [   label,
                # '交換された部品ライフの算術平均 ({:.0f}k, N={})'.format(round(part_life_mean/1000,0), part_life_num),
                # '予防保守の管理目標 ({:.0f}k)'.format(round(params.designed_life * each_wearout_rate/1000,0)),
            ], loc='upper left')

        plt.xlim(params.designed_life * 0.01, params.designed_life * 2.1)
        return parts_exchange_dist

    # 部品交換ヒストグラム (ピンク/ライトブルー)
    def plot_parts_exchange_histogram(each_wearout_rate, xvals, parts_life_data, show_labels=True):
        life_df = pd.DataFrame(
            data = [('予防保守 (打切り)', x) for x in parts_life_data['right_censored']] + [('障害修理', x) for x in parts_life_data['failures']],
            columns=['理由', '累積印刷ページ数']
        )
        # logger.debug(f'life_df=\n{life_df}')
        palette_list = []
        if 1 <= len(life_df.loc[ life_df['理由'] == '予防保守 (打切り)' ]):
            palette_list.append('lightblue')
        if 1 <= len(life_df.loc[ life_df['理由'] == '障害修理' ]):
            palette_list.append('pink')
        # ヒストグラム表示
        sns.histplot(
            data=life_df,
            x='累積印刷ページ数', multiple='stack', hue='理由',
            palette=palette_list,
            linewidth=0.5,
            legend=False
        )
        plt.title('')
        plt.xscale('log')
        if show_labels:
            plt.ylabel(f'部品数')
            plt.legend(title='シミュレーションでの交換理由', loc='center left', labels=['障害修理', '予防保守'], reverse=True)
        else:
            plt.ylabel(f'')

    # 故障確率の算出
    def plot_failure_probability_chart(each_wearout_rate, parts_exchange_dist, parts_strength_dist):
        # 故障確率の算出
        probability_of_failure = my_stress_strength(
            stress = parts_exchange_dist,
            strength = parts_strength_dist,
            warn = False,
            # 以下のパラメータは stress_strength() の改修版でサポートするが、オリジナルでも動作するようラッパー関数 my_stress_strength() を介して呼び出した。詳細は同関数のコメントを参照。
            xlim = (params.designed_life * 0.01, params.designed_life * 2.1),
            stress_label = '応力',
            stress_color = 'red',
            strength_label = '強度',
            strength_color = 'blue',
        )

        logger.debug(f'each_wearout_rate = {each_wearout_rate} probability_of_failure = {probability_of_failure}')
        plt.xscale('log')
        plt.xlim(params.designed_life * 0.01, params.designed_life * 2.1)

        # グラフィック出力
        # --------------------------------
        filename = f'応力-強度干渉グラフ{params.customer_hidden_parameters}({each_wearout_rate:.2f}).png'
        plt.savefig(distination_pathname(params, dt=True, filename=filename))

        return probability_of_failure

    # ---------------------------------------------
    # 強度グラフ + 応力グラフ + 部品交換ヒストグラム作成
    # ---------------------------------------------
    def plot_strength_stress_chart_sub(each_wearout_rate):
        # 分析用データ作成
        parts_life_data = create_analysis_data(each_wearout_rate, result_all_df)
       
        if len(parts_life_data['failures']) < 2:
            return np.nan  # Lognormal_2P は、2件以上の故障データを必要とする

        # X軸区間: 設計値の2倍
        xvals = np.linspace(0, params.designed_life * 2, 200)

        # 1段目 - 部品強度 (青) - parts_strength
        plt.subplot(311)
        parts_strength_dist = plot_strength_chart(each_wearout_rate, xvals)

        # 2段目 - 応力 (赤) - failures_and_survivers
        plt.subplot(312)
        parts_exchange_dist = plot_stress_chart(each_wearout_rate, xvals, parts_life_data)

        # 2段目 - 部品交換ヒストグラム (ピンク/ライトブルー)
        plt.subplot(312).twinx()   # 右側の第2軸を利用する
        plot_parts_exchange_histogram(each_wearout_rate, xvals, parts_life_data)

        # 3段目 - 故障確率の算出
        plt.subplot(313)
        probability_of_failure = plot_failure_probability_chart(each_wearout_rate, parts_exchange_dist, parts_strength_dist)
        return probability_of_failure

    def plot_stress_strength_chart():
        result = []
        for each_wearout_rate in params.wearout_rates:
            fig = init_figure()
            probability_of_failure = plot_strength_stress_chart_sub(each_wearout_rate)
            result.append([each_wearout_rate, probability_of_failure])
            plt.close()
        logger.debug(f'result={result}')
        return result

    result = plot_stress_strength_chart()

    # ---------------------------------------------
    # 故障確率推移グラフ作成
    # ---------------------------------------------
    def plot_failure_probability_trend_chart(result):
        fig = init_figure()
        plt.scatter([x for [x,y] in result], [y for [x,y] in result])
        plt.plot([x for [x,y] in result], [y for [x,y] in result])
        plt.title(f'故障確率 (部品が故障する頻度 [故障数/単位時間]) {params.customer_hidden_parameters}')
        plt.ylim(0.0, 1.0)
        filename = f'故障確率推移グラフ{params.customer_hidden_parameters}.png'
        plt.savefig(distination_pathname(params, dt=True, filename=filename))
        plt.close()

    plot_failure_probability_trend_chart(result)

    # ---------------------------------------------
    # 応力の推移グラフ作成
    # ---------------------------------------------

    # 応力の推移グラフ作成
    # ------------------------
    def plot_stress_trend_chart_sub(each_wearout_rate, primary_ax, secondary_ax, show_labels):
        # 分析用データ作成
        parts_life_data = create_analysis_data(each_wearout_rate, result_all_df)

        # X軸区間: 設計値の2倍
        xvals = np.linspace(0, params.designed_life * 2, 200)

        # 応力のグラフを作成する。ただし Lognormal_2P は、2件以上の故障データを必要とする
        if len(parts_life_data['failures']) >= 2:
            # 応力 (赤) - failures_and_survivers
            plt.axes(primary_ax)   # 左側の第1軸を利用する
            plot_stress_chart(each_wearout_rate, xvals, parts_life_data, show_labels)

        # 部品交換ヒストグラム (ピンク/ライトブルー)
        plt.axes(secondary_ax)   # 右側の第2軸を利用する
        plot_parts_exchange_histogram(each_wearout_rate, xvals, parts_life_data, show_labels)

        # subplot 毎にグラフ上部にタイトル表示
        title_str =(
            '(予防保守の管理目標: 設計値 × 係数 = {:.0f}k'.format(params.designed_life/1000) +
            f' × {each_wearout_rate} = ' +
            '{:.0f}k'.format((params.designed_life * each_wearout_rate)/1000) +
            f') {params.customer_hidden_parameters}'
        )
        plt.title(f'応力-強度モデル {title_str}')

    # 応力の推移グラフ作成
    # --------------------------
    def plot_stress_trend_chart():
        max_rows = 6  # 最大行数
        # logger.debug(f'params.wearout_rates={params.wearout_rates} len(params.wearout_rates)={len(params.wearout_rates)}')

        # 最大行数に収まらない場合、複数の列に分ける
        nrows = min(len(params.wearout_rates), max_rows)            # 行数 (最大 max_rows 行)
        ncols = int((len(params.wearout_rates) - 1) / max_rows) + 1 # 列数
        global fig, axs
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=figsize)
        logger.debug(f'nrows={nrows} ncols={ncols} type(axs)={type(axs)}')

        # 左側の第1軸を作成。便宜のため 1次元の list primary_axs で参照可能にする。
        #   plot.subplots() が返す axe は場合分けする必要があった。
        #     グラフ数 == 1           : Axes           (例) --wearout_rates 1.0
        #     グラフ数 <= 最大行数以下: 1次元 ndarray       --wearout_rates 0.4 0.6 0.8 1.0 1.2 1.4
        #     グラフ数 > 最大行数     : 2次元 ndarray
        if isinstance(axs, matplotlib.axes.Axes):
            # グラフ数が1つの場合
            primary_axs = [axs]    # loop でスキャンできるよう明示的に list にする。
        elif isinstance(axs, np.ndarray) and len(axs.shape) == 1:
            # グラフを1次元に配置する場合
            primary_axs = axs
        elif isinstance(axs, np.ndarray) and len(axs.shape) == 2:
            # グラフを2次元に配置する場合
            i = 0
            j = 0
            primary_axs = []
            for ix, each_wearout_rate in enumerate(params.wearout_rates):
                # logger.debug(f'ix={ix}, each_wearout_rate={each_wearout_rate}, i={i}, j={j}')
                primary_axs.append(axs[i, j])
                j = j + 1 if i == max_rows-1 else j
                i = i + 1 if i < max_rows-1 else 0
        else:
            assert False, f'unexpected: nrows={nrows} ncols={ncols} type(axs)={type(axs)}'
        # logger.debug(f'len(primary_axs)={len(primary_axs)}')

        # 右側の第2軸を作成する。X軸を第1軸と共有する。
        secondary_axs = []
        for ix, ax in enumerate(primary_axs):
            secondary_axs.append(ax.twinx())

        # 第1軸、第2軸について、それぞれsubplotどうしで区間を一致させる
        if len(primary_axs) == 1:
            pass
        else:
            # 第1軸 - subplotどうしで区間を一致させる
            for ix, ax in enumerate(primary_axs):
                # logger.debug(f'ix={ix} ax={str(ax)} --> curr_ax = primary_axs[{ix}] を作成')
                curr_ax = primary_axs[ix]
                if ix == len(primary_axs)-1:
                    # logger.debug(f'ix={ix} ax={str(ax)} --> pass')
                    pass
                else:
                    # logger.debug(f'ix={ix} ax={str(ax)} --> next_ax = primary_axs[{ix+1}] を作成')
                    next_ax = primary_axs[ix+1]
                    # logger.debug(f'ix={ix} ax={str(ax)} --> curr_ax.sharey(next_ax) を実行')
                    curr_ax.sharey(next_ax)

            # 第2軸 - subplotどうしで区間を一致させる
            for ix, ax in enumerate(secondary_axs):
                # logger.debug(f'ix={ix} ax={str(ax)} --> curr_ax = secondary_axs[{ix}] を作成')
                curr_ax = secondary_axs[ix]
                if ix == len(secondary_axs)-1:
                    # logger.debug(f'ix={ix} ax={str(ax)} --> pass')
                    pass
                else:
                    # logger.debug(f'ix={ix} ax={str(ax)} --> next_ax = secondary_axs[{ix+1}] を作成')
                    next_ax = secondary_axs[ix+1]
                    # logger.debug(f'ix={ix} ax={str(ax)} --> curr_ax.sharey(next_ax) を実行')
                    curr_ax.sharey(next_ax)

        matplotlib.rc('font', **font_small)

        # 予防保守の管理目標(係数)ごとに応力のグラフを1枚の画像に表示
        for ix, (each_wearout_rate, primary_ax, secondary_ax) in enumerate(zip(params.wearout_rates, primary_axs, secondary_axs)):
            plot_stress_trend_chart_sub(each_wearout_rate, primary_ax, secondary_ax, True if ix==0 else False)

        filename = f'応力の推移グラフ{params.customer_hidden_parameters}.png'
        plt.savefig(distination_pathname(params, dt=True, filename=filename))
        plt.close()

    plot_stress_trend_chart()

# end-of def show_stress_strength_chart

# 交換部品数と停止時間の棒グラフ作成
def show_summary_graphics(params, result_all_df):
    assert isinstance(result_all_df, pd.DataFrame) and 1 <= len(result_all_df)

    fig = init_figure()
    fig, axes = plt.subplots(2)
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    
    bar_colors = {'予防保守': 'lightblue', '障害修理': 'pink'}

    bar_width = 0.7  # 棒の幅を決める定数

    try_max = result_all_df['try_i'].max()
    logger.debug(f'try_max={try_max}')

    # (1) 交換部品数を算出して棒グラフ表示
    # --------------------------------
    def create_stacked_barchart_for_exchanged_parts_number():
        logger.debug(f'(1) 交換部品数を算出して棒グラフ表示')

        # データ作成
        pm_items = []  # (予防保守)部品数
        cm_items = []  # (障害修理)部品数
        pm_error = []  # (予防保守)エラーバー
        cm_error = []  # (障害修理)エラーバー

        for each_wearout_rate in params.wearout_rates:
            pm_list = []
            cm_list = []
            for i in range(0, try_max + 1):
                try_df = result_all_df.loc[ (result_all_df['管理目標(係数)'] == each_wearout_rate) & (result_all_df['try_i'] == i) ]
                # logger.debug(f'i={i} try_df=\n{try_df}')
                pm_list.append( len(try_df.loc[ try_df['理由'] == '予防保守' ]) )
                cm_list.append( len(try_df.loc[ try_df['理由'] == '障害修理' ]) )

            pm_list = [x for x in pm_list if not np.isnan(x)]
            # logger.debug(f'pm_list={pm_list}')
            cm_list = [x for x in cm_list if not np.isnan(x)]
            # logger.debug(f'cm_list={cm_list}')

            pm_items.append(round(np.mean(pm_list),1))
            cm_items.append(round(np.mean(cm_list),1))

            pm_error.append( np.std(pm_list, ddof=1)/np.sqrt(len(pm_list)) )
            cm_error.append( np.std(cm_list, ddof=1)/np.sqrt(len(cm_list)) )

        exchange_parts = {
            '予防保守': [pm_items, pm_error],
            '障害修理': [cm_items, cm_error],
        }
        logger.debug(f'exchange_parts = {exchange_parts}')

        if len(params.wearout_rates) == 1:
            width = 1.0
        else:
            width = round( (max(params.wearout_rates)-min(params.wearout_rates))/len(params.wearout_rates) * bar_width, 2)
        bottom = np.zeros(len(params.wearout_rates))
        for reason, [exchange_part, y_err] in exchange_parts.items():
            logger.debug(f'(reason, exchange_part, y_err=)={(reason, exchange_part, y_err)}')
            p = axes[0].bar(params.wearout_rates, exchange_part, width, label=reason, bottom=bottom, color=bar_colors[reason], yerr=y_err, ecolor='gray')
            bottom += exchange_part
            axes[0].bar_label(p, label_type='center')
        axes[0].set_title(f'交換部品数\n{simulation_parameters_str(params)}' + (" (エラーバーは標準誤差)" if 1 < params.iter else "") + f' {params.customer_hidden_parameters}' )
        # axes[0].set_xlabel('管理目標(係数)')
        axes[0].set_ylabel('交換部品数')
        axes[0].legend(title="理由", loc='upper right', reverse=True)
    create_stacked_barchart_for_exchanged_parts_number()

    # (2) 停止時間を算出して棒グラフ表示
    # --------------------------------
    def create_stacked_barchart_for_wearout_rates():
        logger.debug('(2) 停止時間を算出して棒グラフ表示')

        # データ作成
        pm_items = []  # (予防保守)停止時間
        cm_items = []  # (障害修理)停止時間
        pm_error = []  # (予防保守)エラーバー
        cm_error = []  # (障害修理)エラーバー

        for each_wearout_rate in params.wearout_rates:
            pm_list = []
            cm_list = []
            for i in range(0, try_max + 1):
                try_df = result_all_df.loc[ (result_all_df['管理目標(係数)'] == each_wearout_rate) & (result_all_df['try_i'] == i) ]
                # logger.debug(f'i={i} try_df=\n{try_df}')
                pm_list.append( try_df.loc[ try_df['理由'] == '予防保守', '停止時間' ].sum() )
                cm_list.append( try_df.loc[ try_df['理由'] == '障害修理', '停止時間' ].sum() )

            pm_list = [x for x in pm_list if not np.isnan(x)]
            # logger.debug(f'pm_list={pm_list}')
            cm_list = [x for x in cm_list if not np.isnan(x)]
            # logger.debug(f'cm_list={cm_list}')

            pm_items.append(round(np.mean(pm_list),1))
            cm_items.append(round(np.mean(cm_list),1))

            pm_error.append( np.std(pm_list, ddof=1)/np.sqrt(len(pm_list)) )
            cm_error.append( np.std(cm_list, ddof=1)/np.sqrt(len(cm_list)) )

        stop_times = {
            '予防保守': [pm_items, pm_error],
            '障害修理': [cm_items, cm_error],
        }
        logger.debug(f'stop_times = {stop_times}')

        if len(params.wearout_rates) == 1:
            width = 1.0
        else:
            width = round( (max(params.wearout_rates)-min(params.wearout_rates))/len(params.wearout_rates) * bar_width, 2)
        bottom = np.zeros(len(params.wearout_rates))
        for reason, [stop_time, y_err] in stop_times.items():
            # logger.debug(f'(reason, stop_time, y_err=)={(reason, stop_time, y_err)}')
            p = axes[1].bar(params.wearout_rates, stop_time, width, label=reason, bottom=bottom, color=bar_colors[reason], yerr=y_err, ecolor='gray')
            bottom += stop_time
            axes[1].bar_label(p, label_type='center')
        axes[1].set_title('停止時間')
        axes[1].set_xlabel('管理目標(係数)')
        axes[1].set_ylabel('停止時間')
        axes[1].legend(title="理由", loc='upper right', reverse=True)
    create_stacked_barchart_for_wearout_rates()

    # (最後) グラフ出力
    # ================================
    # plt.show()
    filename = f'交換部品数と停止時間{params.customer_hidden_parameters}.png'
    plt.savefig(distination_pathname(params, dt=True, filename=filename))
    plt.close()
# end-of def show_summary_graphics

def simulation_entry(params):
    # シミュレーション実行
    def do_simurations():
        # シミュレーション実行
        global end_event

        global wait_times, printing_jobs_log, replacement_parts_log
        wait_times = []             # print_job 毎の印刷所要時間
        printing_jobs_log = []      # print_job 毎の終了時刻と成否
        replacement_parts_log = []  # 交換した部品 [交換理由, 停止時間, 部品情報]

        random.seed(params.seed)      # 乱数発生器の初期化 (デフォルト値 None はシステム時刻使用)

        num_printing_units, num_engineers = [1, 1]

        env = simpy.Environment()  # 環境作成
        env.params = params        # envインスタンスに params 属性を付加。これによりパラメータ引き渡しを簡略化する。
        env.process( printingmachine_simulator_process(env, num_printing_units, num_engineers) )  # 印刷シミュレーションプロセス (平行動作)
        simulation_period = params.maxt    # シミュレーションを期間 params.maxt に渡って行う [単位: 分] デフォルトは1年間

        end_event = env.event()  # シミュレーションを終了させるイベント
        env.run(until=end_event) # シミュレーション実行

        return {
            'wait_times'           : wait_times,             # print_job 毎の印刷所要時間
            'printing_jobs_log'    : printing_jobs_log,      # print_job 毎の終了時刻と成否
            'replacement_parts_log': replacement_parts_log,  # 交換した部品 [交換理由, 停止時間, 部品情報]
        }

    # シミュレーション結果の要約
    def summarize_simulation_results(replacement_parts_log):
        '''シミュレーション結果の要約 (交換理由の別に、停止時間(計画内停止、計画外停止)、交換部品数をサマリー)
        入力
        replacement_parts_log,  # 交換した部品 [交換理由, 停止時間, 部品情報]
        '''
        # ----------------------------------------
        downtime_dict = {
            '予防保守': {'停止時間': 0, '交換部品数': 0},     # 計画内
            '障害修理': {'停止時間': 0, '交換部品数': 0},     # 計画外ダウンタイム
        }
        for item in replacement_parts_log:
            downtime_dict[ item['理由'] ]['停止時間'] += item['停止時間']
            downtime_dict[ item['理由'] ]['交換部品数'] += 1
        # logger.debug(f'    停止時間              : {downtime_dict}')  # downtime_dict = {'予防保守': {'停止時間': 30, '交換部品数': 1}, '障害修理': {'停止時間': 85, '交換部品数': 1}}
        return downtime_dict

    # 印刷シミュレーションを実行して、部品交換リストを返す
    def simulate_each_management_target(wearout_rate):
        '''印刷シミュレーションを実行して、部品交換リスト result_all_df を返す'''
        result_all = []

        for wearout_rate in params.wearout_rates:
            logger.debug(f'wearout_rate={wearout_rate}')

            params.wearout_rate = wearout_rate  # params.wearout_rate であり、params.wearout_rates (末尾s) ではない。両者は別。

            # 同じ条件でシミュレーションを繰り返し、その結果を result_all に追記する
            for try_i in range(params.iter):   # 何回繰り返すか?
                logger.debug(f'  wearout_rate={wearout_rate} try_i={try_i}')

                logger.debug(f'    シミュレーション開始')
                results_dict = do_simurations()               # シミューレションを行う
                downtime_dict = summarize_simulation_results(results_dict['replacement_parts_log'])   # シミュレーション結果を要約
                logger.debug(f'    シミュレーション終了: {downtime_dict}')

                replacement_parts_log = results_dict['replacement_parts_log']
                for item in replacement_parts_log:
                    result_all.append([
                        wearout_rate,     # 管理目標(係数)
                        try_i,            # 繰り返し番号
                        item['T'],        # イベント日時
                        item['理由'],
                        item['停止時間'],
                        item['交換前部品']['交換日時'],  # 交換日時
                        item['交換前部品']['交換理由'],  # 交換理由
                        item['交換前部品']['累積印刷ページ数(ジョブ出力後)']
                    ])
                # end-of for item in replacement_parts_log
            # end-of for try_i in range
        # end-of wearout_rate in 

        result_all_df = pd.DataFrame(result_all, columns=['管理目標(係数)', 'try_i', 'T', '理由', '停止時間', '交換前_交換日時', '交換前_交換理由', '累積印刷ページ数[A4短辺ページ]'])
        return result_all_df

    result_all_df = simulate_each_management_target(params.wearout_rates)
    # logger.debug(f'result_all_df=\n{result_all_df}')

    return result_all_df

def estimation_of_improvement_effects():
    '''保守サービス管理目標の変更による改善効果'''

    def get_params():
        params = Dict()  # Dict() パッケージはドット記法が可能
        params.step           = False
        params.debug          = False
        params.wearout_rates  = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        params.designed_life  = 1000000
        params.beta           = 1.8
        params.eta            = 1000000
        params.check_interval = 60*24*10    # [分]
        params.maxt           = 60*24*30*24 # [分] (= 720.0 [日])
        params.maxx           = 200
        params.iter           = 1
        params.seed           = None
        params.area_coverage  = 'L'
        params.page_length    = 'L'
        params.customer_hidden_parameters = '(AC=' + str(params.area_coverage) + ',PL=' + params.page_length + ')'
        return params

    params = get_params()
    print(f'params={params}')

    global result_all_df

    # シミュレーションとその結果をファイルに保存
    # -------------------------------------
    use_cache = True
    # use_cache = False
    pickle_file = 'result_all_df.pickle'

    if os.path.exists(pickle_file):
        if use_cache:
            with open(pickle_file, 'rb') as f:
                result_all_df = pickle.load(f)
        else:
            result_all_df = simulation_entry(params)
    else:
        result_all_df = simulation_entry(params)

    if use_cache:
        def save_result_all_df(result_all_df):
            with open(pickle_file, 'wb') as f:
                pickle.dump(result_all_df, f)
        save_result_all_df(result_all_df)
            
    def estimate_total_parts_num(a_pair, result_all_df):
        font_small  = {'family' : 'IPAexGothic', 'weight' : 'bold', 'size' : 8}  # 日本語フォント
        plt.figure(figsize=(12,10))
        result_all_df = result_all_df.loc[ result_all_df['管理目標(係数)'].isin(a_pair)].copy()

        # 不要列削除
        for col in ['try_i','理由','停止時間','交換前_交換理由','交換前_交換日時','累積印刷ページ数[A4短辺ページ]','x','y']:
            if col in result_all_df.columns:
                result_all_df.drop([col], axis=1, inplace=True)

        # 'y' 作成
        result_all_df.reset_index(inplace=True, drop=True)
        result_all_df['y'] = result_all_df.index + 1

        def plot_sub(data_df, target_coef, limit_x, formats):
            '''改善前後のデータ点および近似直線の表示'''
            x = data_df['x'].tolist()
            y = data_df['y'].tolist()
            poly = None
            def plot_before_improvement():
                '''改善前のデータ点のプロット'''
                nonlocal poly
                logx = np.log(x)
                logy = np.log(y)
                coeffs = np.polyfit(logx,logy,deg=1)
                poly = np.poly1d(coeffs)
                plt.loglog(x, y, formats[0])
                plt.loglog(x, np.exp(poly(np.log(x))), formats[1])
            plot_before_improvement()

            def plot_after_improvement(): 
                '''改善後のデータ点のプロット'''
                max_x = max(x)
                log_xs = np.linspace(np.log(max_x), np.log(limit_x), num=10)
                xs = [np.exp(x) for x in log_xs]
                ys = [np.exp(poly(x)) for x in log_xs]
                max_y = max(ys)
                plt.loglog(xs, ys, formats[1].replace('-','--'))
                # print(f'max_x = {max_x}, max_y = {max_y}')
                text_label = f'{math.ceil(max_y)}個 (管理目標(係数): {target_coef})'
                plt.text(max(xs), max_y, text_label, fontdict=font_small)
                print(f'target_coef={target_coef} text_label={text_label}')
            plot_after_improvement()

        start_x_cumsum = 1
        limit_x = 60*24*30*48  # X軸のプロット期間: 48ヶ月
        before_or_after = True   # 改善前:True, 改善後:False
        formats = {True: ['ro','r-'], False: ['bx','b-']}

        for target_coef, rest_df in result_all_df.groupby(by=['管理目標(係数)']):
            target_coef = target_coef[0]  # リスト→スカラー
            rest_df['x'] = rest_df['T'] + start_x_cumsum
            result_all_df.loc[ result_all_df['y'].isin(rest_df['y']), 'x'] = rest_df['x']
            max_x = rest_df['x'].max()

            # 改善前後のデータ点および近似直線の表示
            plot_sub(
                data_df     = rest_df,
                target_coef = target_coef,
                limit_x     = limit_x,
                formats     = formats[before_or_after],
            )
            start_x_cumsum = max_x
            before_or_after = False

        # 縦補助線 (1, 2, 3, 4年目)
        xlims = plt.xlim()
        ylims = plt.ylim()
        duration = 0
        for tick_x in [x* 60*24*30 for x in range(0, int(limit_x/(60*24*30)) + 1, 12)]:
            plt.vlines(tick_x, ylims[0], ylims[1], linestyle=':', color='lightgray', zorder=0)
            plt.text(tick_x, ylims[1], f'{duration}年目')
            duration += 1

        plt.xlim([8e4, xlims[1]])
        plt.ylim([1  , 1e2])
        plt.title('サービス中の保守変更 (定期交換部品のライフ目標値変更)\n')
        # plt.show()

        filename = f'保守サービス管理目標の変更による改善効果({a_pair}){params.customer_hidden_parameters}.png'
        plt.savefig(distination_pathname(params, dt=True, filename=filename))
        plt.close()

    # 比較したい管理目標(係数)の組
    wearout_rate_pairs = [
        [0.7, 0.8],
        [0.7, 0.9],
        [0.7, 1.0],
        [0.8, 0.9],
        [0.8, 1.0],
        [0.8, 1.1],
        [0.9, 1.0],
        [0.9, 1.1],
        [0.9, 1.2],
    ]
    for a_pair in wearout_rate_pairs:
        estimate_total_parts_num(a_pair, result_all_df)

def main():
    params = arg_parse()

    # 本スクリプトを保存 (再現性を高める)
    for file in [this_file, import_file]:
        if os.path.exists(file):
            shutil.copyfile(file, distination_pathname(params, dt=True, filename=file))

    init_logging(logfile=distination_pathname(params, dt=True, filename='debug.log'))
    logger.debug('start')
    logger.debug(f'args={args}')  # コマンドライン引数を記録
    logger.debug(f'params={params}')

    # estimates_improvement_effects = True    # 複数のシミュレーションにより、保守サービス管理目標の変更による改善効果を推定する
    estimates_improvement_effects = False  # 単一のシミュレーションのみ行う。パラメータはコマンドラインで指定。

    if estimates_improvement_effects:
        # 保守サービス管理目標の変更による改善効果
        estimation_of_improvement_effects()
    else:
        # 単一のシミュレーションのみ行う。パラメータはコマンドラインで指定。
        global result_all_df
        result_all_df = simulation_entry(params)

        # 応力-強度干渉グラフ作成
        show_stress_strength_chart(params, result_all_df)

        # 交換部品数と停止時間の棒グラフ作成
        show_summary_graphics(params, result_all_df)

        def show_parameters(parms):
            '''シミュレーション実行条件を表示'''
            result = \
                f'args={args}' + '\n'\
                f'予防保守の管理目標(係数)  args.wearout_rates  = {args.wearout_rates} ({params.wearout_rates})' + '\n'\
                f'部品ライフ設計値          args.designed_life  = {args.designed_life} [A4短辺ページ]' + '\n'\
                f'部品ライフ形状パラメータ  args.beta           = {args.beta}' + '\n'\
                f'部品ライフ尺度パラメータ  args.eta            = {args.eta}' + '\n'\
                f'保守間隔                  args.check_interval = {args.check_interval} [分] (= {params.check_interval} [分] = {params.check_interval/(60*24)} [日])' + '\n'\
                f'シミュレーション期間      args.maxt           = {args.maxt} [分] (= {params.maxt/(60*24)} [日])' + '\n'\
                f'交換部品数の最大値        args.maxx           = {args.maxx}' + '\n'\
                f'シミュレーション回数      args.iter           = {args.iter}' + '\n'\
                f'random.seed() 初期値      args.seed           = {args.seed} ({args.seed if args.seed else "システム時刻使用"})' + '\n'\
                f'area_coverage             args.area_coverage  = {args.area_coverage}' + '\n'\
                f'page_length               args.page_length    = {args.page_length}'

            return result
        logger.debug(show_parameters(params))

    logger.debug('successfully completed')
    logging.shutdown()
# end-of def main

if __name__ == '__main__':
    main()
