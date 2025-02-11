# !python

# sim_hidden_param.py - サマライズされた印刷機の稼働情報から、元の印刷ジョブを推定する
# Written in 2025 by yasuakih

'''
usage: sim_hidden_param.py [-h] [--iterations ITERATIONS] [--printing_machines [PRINTING_MACHINES ...]] [--pickle PICKLE] [--cpu_count CPU_COUNT]

options:
  -h, --help            show this help message and exit
  --iterations ITERATIONS
  --printing_machines [PRINTING_MACHINES ...]
  --pickle PICKLE
  --cpu_count CPU_COUNT

トラブルシューティング
TclError: invalid command name ".!navigationtoolbar2tk.!button2"

'''

import sys
import random
import numpy as np
# import simpy
import reliability
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors
import japanize_matplotlib
import seaborn as sns
import statistics
import math
import argparse
import pandas as pd
import os
import gzip
import pickle
import cProfile
import shutil
import multiprocessing as mp
import logging
from datetime import datetime
from math import log2
from collections import Counter
from reliability.Distributions import Gamma_Distribution, Normal_Distribution
from reliability.Fitters import Fit_Gamma_2P, Fit_Weibull_2P, Fit_Normal_2P
from reliability.Other_functions import histogram
import importlib

this_file = 'sim_hidden_param.py'  # このファイルのファイル名
import_file = ''  # ロジックとデータの分離。ファイルがなくても本スクリプトは動作する。
import_file_var = None

# ----------------------------------------------------------------
# ユーティリティ
# ----------------------------------------------------------------

def arg_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--debug', action='store_true', default=False)
    # parser.add_argument(
    #     '--show', action='store_true', default=False,
    #     help='--showを指定するとグラフを画面に表示する。(デフォルト: False → ファイルに保存)'
    # )
    parser.add_argument(
        '--iterations', type=int, default=1,
        help='モンテカルロ法の総実行回数を指定する (デフォルト: 1)。 実用上の上限は 1000 程度。 (例: --iterations 1000)')
    parser.add_argument(
        '--printing_machines', action='extend', nargs='*', default=[],
        help='シミュレーション対象の印刷機の名前 (例: --printing_machines PM1 PM2)')
    parser.add_argument(
        '--pickle', type=str, default=None,
        help='既存のシミュレーション結果 pickle ファイルを指定する。' +
             'この場合、シミュレーションは行わなずに、シミュレーション結果だけを表示する。' +
             '大域変数 sim_result_all に格納されたデータはデバッグ時に生かせる。' +
             'マルチプロセスの場合は機能しない制約がある。(例: --pickle )')
    parser.add_argument(
        '--cpu_count', type=int, default=1,
        help='シミュレーションをマルチプロセスで行う場合、使用するCPU数を指定する。' +
             '1=シングルプロセス(デフォルト), 2以上=マルチプロセス)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random.seed() を指定する。0 の場合はシステム時刻を使う (デフォルト: 42)。 (例: --seed 0)')
    parser.add_argument(
        '--import_file', type=str, default='',
        help='印刷機のデータファイル')

    args = parser.parse_args()

    # --import_file foo.py あるいは --import_file foo が指定された場合、大域変数 import_file <= 'foo.py' をストアする。
    # それ以外の場合、import_file は '' をとる。

    global import_file
    global import_file_var
    # print(f'args.import_file = {args.import_file} ({type(args.import_file)})')
    # print(f'import_file = {import_file} ({type(import_file)})')
    if args.import_file == '':
        # import_file = ''
        pass
    else:
        if args.import_file[-3:] == '.py':
            import_file = args.import_file
        else:
            import_file = args.import_file + '.py'
        assert os.path.exists(import_file), f'file not found: {import_file}'
        import_file_var = importlib.import_module(import_file[0:-3])  # ファイル名から '.py' を除いてモジュール名を得、インポートする
    print(f'import_file = {import_file} ({type(import_file)})')

    return args
# end-of def arg_parse

args = arg_parse()
print(args)

def single_processing():
    try:
        if args.cpu_count == 1:
            return True
        else:
            return False
    except NameError:
        return False
# end-of def single_processing

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

def logger_debug(s):
    '''multiprocessing で生成した子プロセスは、親プロセスの logger と上手く共存できなかったので、その場合は単純な print 出力で代用した'''
    if single_processing():
        logger.debug(s)
    else:
        print(s)
# end-of def logger_debug

def my_gauss(mu, sigma, upper_limit, number_of_digits):
    '''離散的なガウス分布を生成する
    lower_limit 下限値
    upper_limit 上限値
    number_of_digits 小数点以下の桁数
      0: ページ長で用いる (小数点以下は切り捨て)
      2: カバレッジと両面比で用いる (小数点以下 2桁を残して切り捨て)
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
'''
plt.close()
l = []
for i in range(1, 10000):
    v = my_gauss(0.20, 0.10, 0.8, 2)
    l.append(v)
sns.histplot(l, bins=100,  binrange=[0.0,1.0])
plt.show()

plt.close()
l = []
for i in range(1, 10000):
    v = my_gauss(200, 300, 2000, 0)
    l.append(v)
sns.histplot(l, bins=100, binrange=[0,2000])
plt.show()
'''

savedir_path = None
def distination_pathname(data_pathname='_debug', dt=False, filename='john.doe'):
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
        savedir_path = os.path.join(data_pathname, d_str)

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

def init_figure():
    '''新しい図を作成'''
    # Fail to allocate bitmap エラー回避
    # if args.show:
    #     pass
    # else:
    #     matplotlib.use("Agg")
    matplotlib.use("Agg")

    fig = plt.figure(figsize=(14, 12))  # (横, 縦)
    font = {'family' : 'IPAexGothic', 'weight' : 'bold', 'size' : 7}  # 日本語フォント。サイズは小さめ。
    matplotlib.rc('font', **font)
    return fig
# end-of def init_figure

def page_size_str(page_size, prefix=False):
    '''用紙サイズにサイズ順の数字を付加する'''
    assert page_size in ['葉書', 'A4', 'B4', 'A3', '長尺'], f'page_size = {page_size}'

    if   page_size == '葉書':     pf = '1';        result = '葉書'
    elif page_size == 'A4':       pf = '2';        result = 'A4'
    elif page_size == 'B4':       pf = '3';        result = 'B4'
    elif page_size == 'A3':       pf = '4';        result = 'A3'
    elif page_size == '長尺':     pf = '5';        result = '長尺'
    else:                         pf = '6';        result = f'{page_size}'
    if prefix:
        return pf + ' ' + result
    else:
        return result
# end-of def page_size_str        

def page_length_range_str(page_length, prefix=False):
    if      1 <= page_length <= 10:        pf = '00';        ix = '1-10'
    elif   11 <= page_length <= 20:        pf = '01';        ix = '11-20'
    elif   21 <= page_length <= 30:        pf = '02';        ix = '21-30'
    elif   31 <= page_length <= 50:        pf = '03';        ix = '31-50'
    elif   51 <= page_length <= 100:       pf = '04';        ix = '51-100'
    elif  101 <= page_length <= 200:       pf = '05';        ix = '101-200'
    elif  201 <= page_length <= 300:       pf = '06';        ix = '201-300'
    elif  301 <= page_length <= 500:       pf = '07';        ix = '301-500'
    elif  501 <= page_length <= 1000:      pf = '08';        ix = '501-1000'
    elif 1001 <= page_length <= 2000:      pf = '09';        ix = '1001-2000'
    else:                                  pf = '10';        ix = '2001'
    if prefix:
        return pf + ' ' + ix
    else:
        return ix
# end-of def page_length_range_str

if os.path.exists(import_file):
    page_length_range_str = import_file_var.page_length_range_str

def area_coverage_range_str(area_coverage, prefix=False):
    '''オフセット印刷では運用上、80%/色 を超えない'''
    if   0.00 <= area_coverage < 0.05:       pf = '00';        ac = '0-5%'
    elif 0.05 <= area_coverage < 0.10:       pf = '01';        ac = '5-10%'
    elif 0.10 <= area_coverage < 0.15:       pf = '02';        ac = '10-15%'
    elif 0.15 <= area_coverage < 0.20:       pf = '03';        ac = '15-20%'
    elif 0.20 <= area_coverage < 0.25:       pf = '04';        ac = '20-25%'
    elif 0.25 <= area_coverage < 0.30:       pf = '05';        ac = '25-30%'
    elif 0.30 <= area_coverage < 0.35:       pf = '06';        ac = '30-35%'
    elif 0.35 <= area_coverage < 0.40:       pf = '07';        ac = '35-40%'
    elif 0.40 <= area_coverage < 0.45:       pf = '08';        ac = '40-45%'
    elif 0.45 <= area_coverage < 0.50:       pf = '09';        ac = '45-50%'
    elif 0.50 <= area_coverage < 0.55:       pf = '10';        ac = '50-55%'
    elif 0.55 <= area_coverage < 0.60:       pf = '11';        ac = '55-60%'
    elif 0.60 <= area_coverage < 0.65:       pf = '12';        ac = '60-65%'
    elif 0.65 <= area_coverage < 0.70:       pf = '13';        ac = '65-70%'
    elif 0.70 <= area_coverage < 0.75:       pf = '14';        ac = '70-75%'
    elif 0.75 <= area_coverage < 0.80:       pf = '15';        ac = '75-80%'
    else:                                    pf = '16';        ac = f'{area_coverage}'
    if prefix:
        return pf + ' ' + ac
    else:
        return ac
# end-of area_coverage_range_str

# 交差エントロピー (Cross-entropy) による2つの確率分布の類似性の定量化
def cross_entropy(p, q, events = None):
    '''
    交差エントロピー (Cross-entropy) による2つの確率分布の類似性の定量化
    (参照)
    https://ja.wikipedia.org/wiki/%E4%BA%A4%E5%B7%AE%E3%82%A8%E3%83%B3%E3%83%88%E3%83%AD%E3%83%94%E3%83%BC
    https://en.wikipedia.org/wiki/Cross-entropy
    https://github.com/PacktPublishing/Hands-On-Simulation-Modeling-with-Python-Second-Edition/blob/main/Chapter04/cross_entropy.py
    '''

    def replace_0_in_list_with_1(a_list):
        # list a_list に含まれる要素が 0 の場合、1 へ置換する。log2(0) は定義されないため本処理を適用した。
        l = []
        for item in a_list:
            if item == 0:
                l.append(1)
            else:
                l.append(item)
        return l

    p = replace_0_in_list_with_1(p)  # 要素が 0 の場合は計算できないため、便宜上、(0に近い意味で) 1 へ置換する。
    q = replace_0_in_list_with_1(q)  # 

    p = [item/sum(p) for item in p]  # 分布の合計を1.0とする
    q = [item/sum(q) for item in q]  # 

    h_pq = -sum([p*log2(q) for p,q in zip(p,q)]) # 公差エントロピーの計算

    if events:
        # 新しい図を作成
        fig = init_figure()

        plt.subplot(2,1,1)
        plt.bar(events, p)
        plt.title('p')

        plt.subplot(2,1,2)
        plt.bar(events, q)
        plt.title('q')

        plt.suptitle(f'cross entropy = {h_pq:.2f} bit')
        plt.show()

        f.clear()
        plt.close(f)

    return h_pq
'''
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from math import log2
p = [0, 1000, 2000, 3000, 10]
q = [0, 1100, 2800, 2000, 0]  # 1.563 bit
# q = [0, 2000, 2800, 1100, 0]  # 1.850 bit
h_pq = cross_entropy(p, q)
print(f'{h_pq:.3f} bit')
'''

# ----------------------------------------------------------------
# クラス
# ----------------------------------------------------------------

# 顧客のオンデマンド印刷の特徴
# --------------------------------
class Customer():

    # 閾値は経験的に決めた
    OK_NG_BORDER = {
        'h1_ce_lim' : 1.0,
        'h2_ce_lim' : 3.3,
    }

    if os.path.exists(import_file):
        OK_NG_BORDER = import_file_var.OK_NG_BORDER

    customer_type     = None  # オンデマンド印刷機の種類
    area_coverage_lvl = None  # オンデマンド印刷機のエリアカバレッジの程度
    page_length_lvl   = None  # オンデマンド印刷機の印刷ページ長の程度

    CUTOMER_TYPE = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']  # 顧客のオンデマンド印刷の特徴を説明する仮説を、計9通り作成する
    CUTOMER_DICT = {
        'Q1': {'area_coverage_lvl': 'L', 'page_length_lvl': 'L'},
        'Q2': {'area_coverage_lvl': 'L', 'page_length_lvl': 'M'},
        'Q3': {'area_coverage_lvl': 'L', 'page_length_lvl': 'H'},
        'Q4': {'area_coverage_lvl': 'M', 'page_length_lvl': 'L'},
        'Q5': {'area_coverage_lvl': 'M', 'page_length_lvl': 'M'},
        'Q6': {'area_coverage_lvl': 'M', 'page_length_lvl': 'H'},
        'Q7': {'area_coverage_lvl': 'H', 'page_length_lvl': 'L'},
        'Q8': {'area_coverage_lvl': 'H', 'page_length_lvl': 'M'},
        'Q9': {'area_coverage_lvl': 'H', 'page_length_lvl': 'H'},
    }

    def generate_customer_printed_distribution(self):
        '''顧客のオンデマンド印刷の特徴をランダムに作成。これを元に印刷ジョブが作成される。'''

        # (1) オンデマンド印刷機の印刷機セグメントを仮定
        self.customer_type     = random.sample(self.CUTOMER_TYPE, 1)[0]    # 仮説を無作為に1つ選択。[0]は返り値のlist型からstr型にするため。
        self.area_coverage_lvl = self.CUTOMER_DICT[self.customer_type]['area_coverage_lvl']  # トータルエリアカバレッジ [L, M, H] の3段階
        self.page_length_lvl   = self.CUTOMER_DICT[self.customer_type]['page_length_lvl']    # 印刷ページ長 [L, M, H] の3段階

        # logger_debug(f'  class Customer(): self.customer_type={self.customer_type} self.area_coverage_lvl={self.area_coverage_lvl} self.page_length_lvl={self.page_length_lvl}')

        # (2) 印刷用紙のサイズ別割合を仮定
        def split_job_by_paper_sizes():
            printed_matter = PrintedMatter(None, None)  # このインスタンスは、インスタンス変数の参照用として用いる
            total_printed_matters = len(printed_matter.PRINTED_MATTERS.keys())  # 利用可能な印刷用紙の種類の数

            # 使用する印刷用紙の種類(の数)。一様乱数で決める。
            customer_printed_matters_min = int(total_printed_matters * 0.6)  # 経験的な値
            customer_printed_matters_max = int(total_printed_matters * 0.8)
            customer_printed_matters = random.randint(customer_printed_matters_min, customer_printed_matters_max)  # お客様がサポートする印刷物の種類の数。[customer_printed_matters_min, customer_printed_matters_max] と仮定。
            # logger_debug(f'customer_printed_matters = {customer_printed_matters}')

            split_points = sorted(random.sample(range(1, 99), customer_printed_matters - 1)) # それぞれの割合をランダムに配置。[1-99] までの区間を分割。
            # logger_debug(f'split_points = {split_points}')

            self.customer_printed_matters_list = random.sample(list(printed_matter.PRINTED_MATTERS.keys()), customer_printed_matters)  # その種類をランダムに列挙
            # logger_debug(f'self.customer_printed_matters_list = {self.customer_printed_matters_list}')

            # 顧客の印刷機に固有のオンデマンド印刷物の特徴を作成。これを元に印刷ジョブが作成される。
            seg = {}
            if customer_printed_matters == 1:
                # seg = {}
                seg[ self.customer_printed_matters_list[0] ] = 100
            else:
                # 区間 [0-100] をランダムに分割する。 (101要素あるのに注意)
                segments = []
                segments.append( [0, split_points[0]] )  # 最初の区間 (0から始まる)
                for i in range(1, len(split_points)):
                    segments.append( [split_points[i-1], split_points[i]] )
                segments.append( [split_points[len(split_points)-1], 100] )  # 最後の区間 (100で終わる)
                # logger_debug(f'segments = {segments}')
                # seg = {}
                for i, key in enumerate(self.customer_printed_matters_list):
                    seg[key] = segments[i][1] - segments[i][0]
            # logger_debug(f'seg = {seg}')

            # check-1 合計は 100 になる
            assert sum( [seg[key] for key in list(seg.keys())] ) == 100, f'seg = {seg}'

            # check-2 お客様がサポートする印刷物の割合は 0 を超え、印刷物の数は [customer_printed_matters_min, customer_printed_matters_max] のいずれか。
            not_zero = 0
            for item in list(seg.keys()):
                if 0.0 < float(seg[item]):
                    not_zero += 1
            assert customer_printed_matters_min <= not_zero <= customer_printed_matters_max, f'not_zero={not_zero} segments={segments} seg={seg}'

            return seg
        # end-of def split_job_by_paper_sizes()

        self.matters = split_job_by_paper_sizes()
        # logger_debug(f'  class Customer(): self.matters = {self.matters}')

        # if (__name__=='__main__') and args.debug:
        #     logger_debug(f'  class Customer(): self.customer_type={self.customer_type} self.area_coverage_lvl={self.area_coverage_lvl} self.page_length_lvl={self.page_length_lvl} self.matters={self.matters}')

    def __init__(self):
        self.generate_customer_printed_distribution()

        # logger.debug(f'self.OK_NG_BORDER={self.OK_NG_BORDER}')
        self.h1_ce_lim = self.OK_NG_BORDER['h1_ce_lim']
        self.h2_ce_lim = self.OK_NG_BORDER['h2_ce_lim']

    def __str__(self):
        printed_matter = PrintedMatter(None, None)  # このインスタンスは、インスタンス変数の参照用として用いる
        result_s = ''
        for item in list(printed_matter.PRINTED_MATTERS.keys()):
            if item in self.customer_printed_matters_list:
                v = self.matters[item]
            else:
                v = 0
            # logger_debug(f'{item} : {v}')
            result_s += item + ' ' + f'{v:>2}' + ' '
        result_s = '[' + result_s.rstrip(' ').replace('長尺','長').replace('葉書','葉') + ']'
        return f'[TYP={self.customer_type} AC={self.area_coverage_lvl} PL={self.page_length_lvl}]  {result_s}'
# end-of class Customer()


# オンデマンド印刷物の印刷ジョブ
# --------------------------------
class PrintedMatter():
    '''オンデマンド印刷における典型的な、TAC値、用紙サイズ、ページ数、両面比率を仮定した。'''

    MAX_PAGE_LENGTH = 2000  # 印刷ページ長の最大 (最小は1)
    MAX_SET_PER_JOB = 2000  # 印刷部数の最大 (最小は1)

    PRINTED_MATTERS = {
        '葉書': None,
        'A4' : None,
        'B4' : None,
        'A3' : None,
        '長尺': None,
    }

    # 各パラメータの整合性をチェック
    def check_for_consistents(self):
        # エリアカバレッジ整合性
        for matter in self.PRINTED_MATTERS.keys():
            area_coverage_min, area_coverage_max = self.PRINTED_MATTERS[matter]['area_coverage']
            assert area_coverage_min <= area_coverage_max, f'matter={matter} area_coverage_min={area_coverage_min} area_coverage_max={area_coverage_max}'
            assert 0.0 < area_coverage_min <= 1.0, f'matter={matter} area_coverage_min={area_coverage_min}'
            assert 0.0 < area_coverage_max <= 1.0, f'matter={matter} area_coverage_max={area_coverage_max}'

        # 用紙サイズ整合性
        papers = Paper().PAPERS
        for matter in self.PRINTED_MATTERS.keys():
            for paper_size in self.PRINTED_MATTERS[matter]['paper_size']:
                assert paper_size in papers, f'matter={matter} paper_size={paper_size}'

        # 印刷ページ長整合性
        for matter in self.PRINTED_MATTERS.keys():
            page_length_min, page_length_max = self.PRINTED_MATTERS[matter]['page_length']
            assert page_length_min <= page_length_max, f'matter={matter} page_length_min={page_length_min} page_length_max={page_length_max}'
            assert 1 <= page_length_min <= self.MAX_PAGE_LENGTH, f'matter={matter} page_length_min={page_length_min}'
            assert 1 <= page_length_max <= self.MAX_PAGE_LENGTH, f'matter={matter} page_length_max={page_length_max}'

        # 印刷部数整合性
        # for matter in self.PRINTED_MATTERS.keys():
        #     set_per_job_min, set_per_job_max = self.PRINTED_MATTERS[matter]['set_per_job']
        #     assert set_per_job_min <= set_per_job_max, f'matter={matter} set_per_job_min={set_per_job_min} set_per_job_max={set_per_job_max}'
        #     assert 1 <= set_per_job_min <= self.MAX_SET_PER_JOB, f'matter={matter} set_per_job_min={set_per_job_min}'
        #     assert 1 <= set_per_job_max <= self.MAX_SET_PER_JOB, f'matter={matter} set_per_job_max={set_per_job_max}'

        # 両面比整合性
        for matter in self.PRINTED_MATTERS.keys():
            duplex_rate_min, duplex_rate_max = self.PRINTED_MATTERS[matter]['duplex_rate']
            assert duplex_rate_min <= duplex_rate_max, f'matter={matter} duplex_rate_min={duplex_rate_min} duplex_rate_max={duplex_rate_max}'
            assert 0.0 <= duplex_rate_min <= 1.0, f'matter={matter} duplex_rate_min={duplex_rate_min}'
            assert 0.0 <= duplex_rate_max <= 1.0, f'matter={matter} duplex_rate_max={duplex_rate_max}'

        # カラー比整合性
        # for matter in self.PRINTED_MATTERS.keys():
        #     color_rate_min, color_rate_max = self.PRINTED_MATTERS[matter]['color_rate']
        #     assert color_rate_min <= color_rate_max, f'matter={matter} color_rate_min={color_rate_min} color_rate_max={color_rate_max}'
        #     assert 0.0 <= color_rate_min <= 1.0, f'matter={matter} color_rate_min={color_rate_min}'
        #     assert 0.0 <= color_rate_max <= 1.0, f'matter={matter} color_rate_max={color_rate_max}'

        pass
    # end-of def check_for_consistents

    def local_debug_print(self, s):
        # logger_debug(s)
        pass
    # end-of def local_debug_print

    def __init__(self, customer, printing_machine):
        # try:
        #     # --debug 指定時、パラメータ整合性をチェック
        #     if (__name__=='__main__') and args.debug:
        #         self.check_for_consistents()
        # except NameError:
        #     pass

        # トータルエリアカバレッジ - 用紙サイズ別に仮定
        area_coverage_list = {
            '葉書': {'L': [0.10, 0.10], 'M': [0.20, 0.10], 'H': [0.20, 0.20]},
            'A4'  : {'L': [0.03, 0.05], 'M': [0.10, 0.10], 'H': [0.20, 0.20]},
            'B4'  : {'L': [0.03, 0.05], 'M': [0.04, 0.10], 'H': [0.05, 0.20]},
            'A3'  : {'L': [0.03, 0.05], 'M': [0.04, 0.10], 'H': [0.05, 0.20]},
            '長尺': {'L': [0.10, 0.10], 'M': [0.30, 0.10], 'H': [0.50, 0.20]},
        }

        # 印刷ページ長 - 用紙サイズ別に仮定
        page_length_list = {
            '葉書': {'L': [0,  50], 'M': [  0, 100], 'H': [  0, 200]},
            'A4'  : {'L': [0, 300], 'M': [300, 300], 'H': [500, 600]},
            'B4'  : {'L': [0, 300], 'M': [300, 300], 'H': [500, 600]},
            'A3'  : {'L': [0, 300], 'M': [300, 300], 'H': [500, 600]},
            '長尺': {'L': [0,   5], 'M': [  0,  10], 'H': [  0,  30]},
        }
        '''
# 上述の分布をグラフ表示する。
# iPythonを起動し、スクリプトをロード (--help)、2つの変数をコピーペーストでglobalに定義し、このコメントブロック内のコードをコピーペーストして実行する
# %run sim_hidden_param.py --help
# -- 8< -- 8< -- 8< -- 8< -- 8< --
plt.close()
plt.figure(figsize=(8, 8))
# area_coverage_list を表示
for i, paper_size in enumerate([item for item in list(area_coverage_list.keys()) if item in ['葉書','A4','B4','A3','長尺']]):
    plt.subplot(2,5,i+1)
    for level in ['L','M','H']:
        print(f'paper_size={paper_size} level={level}')
        l = []
        for i in range(1, 10000):
            v = my_gauss(area_coverage_list[paper_size][level][0], area_coverage_list[paper_size][level][1], 0.8, 2)
            l.append(v)
        sns.histplot(l, bins=50, binrange=[0.0, 1.0])
        plt.title(paper_size)

# page_length_list を表示
for i, paper_size in enumerate([item for item in list(page_length_list.keys()) if item in ['葉書','A4','B4','A3','長尺']]):
    plt.subplot(2,5,i+1+5)
    for level in ['L','M','H']:
        print(f'paper_size={paper_size} level={level}')
        l = []
        for i in range(1, 100000):
            v = my_gauss(page_length_list[paper_size][level][0], page_length_list[paper_size][level][1], 2000, 0)
            l.append(v)
        sns.histplot(l, bins=100, binrange=[0,1000])
        # sns.histplot(l, log_scale=True)  # X軸を対数表示すると離散的なバーが櫛歯のように表示されて見づらい
        plt.title(paper_size)
plt.show()
# -- 8< -- 8< -- 8< -- 8< -- 8< --
        '''

        # 両面比 - 用紙サイズ別に仮定
        duplex_rate_list = {
            '葉書': [0.5, 0.3],
            'A4'  : [0.5, 0.3],
            'B4'  : [0.5, 0.3],
            'A3'  : [0.5, 0.3],
            '長尺': [0.5, 0.3],
        }

        if customer is None:
            # __init__() 関数内のローカル変数を外部から参照できるようにしたい。
            # 仮引数 customer が None の場合、ローカル変数をインスタンス変数として参照可能にして処理を終える。
            self.area_coverage_list = area_coverage_list
            self.page_length_list = page_length_list
            self.duplex_rate_list = duplex_rate_list
            pass
        else:
            self.local_debug_print(f'    class PrintedMatter(): customer.customer_type={customer.customer_type} customer.area_coverage_lvl={customer.area_coverage_lvl} customer.page_length_lvl={customer.page_length_lvl}')

            # 1. 印刷物の特徴
            #    (呼び出し元から与えられた) 顧客の印刷機に固有のオンデマンド印刷物の特徴を得る。
            customer_printed_matters = customer.matters
            self.local_debug_print(f'    class PrintedMatter(): customer_printed_matters = {customer_printed_matters}')
            assert isinstance(customer_printed_matters, dict), f'type(customer_printed_matters) = {type(customer_printed_matters)}'

            #    その特徴に基づき、印刷用紙のサイズを無作為に決める。
            #    用紙サイズが決まると後続のエリアカバレッジ、用紙サイズ、印刷ページ長、両面比の統計的な分布も決まる。
            matter = random.choices(
                list(customer_printed_matters.keys()),
                weights = list(customer_printed_matters.values())
            )[0]
            self.type = matter
            self.local_debug_print(f'customer_printed_matters={customer_printed_matters} type={type}')
            # self.local_debug_print(f'    class PrintedMatter(): matter = {matter} (ランダム)')

            # 2. エリアカバレッジ
            area_coverage_mu, area_coverage_sigma = area_coverage_list[matter][customer.area_coverage_lvl]
            self.area_coverage = my_gauss(area_coverage_mu, area_coverage_sigma, 0.80, 2)
            self.local_debug_print(f'    class PrintedMatter(): self.area_coverage={self.area_coverage:.2f}')
            # logger_debug(f'    class PrintedMatter(): matter={matter}\tcustomer.area_coverage_lvl={customer.area_coverage_lvl}\tmu={area_coverage_mu}\tsigma={area_coverage_sigma}\tarea_coverage={self.area_coverage}')

            # 3. 用紙サイズ
            self.paper_size = matter

            # 4. 印刷ページ長
            page_length_mu, page_length_sigma = page_length_list[matter][customer.page_length_lvl]
            self.page_length = int(my_gauss(page_length_mu, page_length_sigma, self.MAX_PAGE_LENGTH, 0))
            self.local_debug_print(f'    class PrintedMatter(): self.page_length={self.page_length}')
            # logger_debug(f'    class PrintedMatter(): matter={matter}\tcustomer.page_length_lvl={customer.page_length_lvl}\tmu={page_length_mu}\tsigma={page_length_sigma}\tpage_length={self.page_length}')

            # 5. 両面比
            duplex_rate_mu, duplex_rate_sigma = duplex_rate_list[matter]
            self.duplex_rate = my_gauss(duplex_rate_mu, duplex_rate_sigma, 1.0, 2)
            if 0.5 <= self.duplex_rate:
                self.duplex_or_simplex = 'duplex'
            else:
                self.duplex_or_simplex = 'simplex'
            # self.local_debug_print(f'    class PrintedMatter(): self.duplex_or_simplex={self.duplex_or_simplex}')

            # 6. カラー比
            # color_rate_min, color_rate_max = self.PRINTED_MATTERS[matter]['color_rate']
            # self.color_rate = float(int(random.uniform(color_rate_min, color_rate_max)*100)/100)
            # if 0.5 <= self.color_rate:
            #     self.color_or_mono = 'color'
            # else:
            #     self.color_or_mono = 'mono'
        # end-of if customer is None
    # end-of def __init__

    def __str__(self):
        return [
            str(self.type) + ', ' + 
            str(self.area_coverage) + ', ' + 
            str(self.paper_size) + ', ' + 
            str(self.page_length) + 'p, ' + 
            # str(self.set_per_job) + 'job, ' + 
            str(self.duplex_or_simplex)
            # str(self.color_or_mono)
        ][0]
    # end-of def __str__
# end-of class PrintedMatter()

# 用紙
# --------------------------------
class Paper():

    # 用紙面積 [m2]
    PAGE_AREA_IN_M2 = {
        '葉書'  : 0.0148,
        'A4'    : 0.06237,
        'B4'    : 0.093548,
        'A3'    : 0.159,
        '長尺'  : 0.4,
    }

    if os.path.exists(import_file):
        PAGE_AREA_IN_M2 = import_file_var.PAGE_AREA_IN_M2

    def __init__(self, size = None):

        self.PAPERS = list(self.PAGE_AREA_IN_M2.keys())

        if size == None:
            pass
        else:
            assert size in self.PAGE_AREA_IN_M2.keys(), f'size = {size}'
            self.size = size
            self.page_area = self.PAGE_AREA_IN_M2[size]

    def __str__(self):
        return f'self.size={self.size}\tself.page_area={self.page_area}'

    def size_str(self):
        return self.size + '    ' if self.size in ['A4', 'B4', 'A3'] else self.size
# end-of Class Paper()

# トータルエリアカバレッジ(TAC値)
# --------------------------------
class AreaCoverage():
    AREA_COVERAGE_RANGE_DICT = {
        '0-5%'   : [0.00 , 0.05, '00'],
        '6-10%'  : [0.06 , 0.10, '01'],
        '11-15%' : [0.11 , 0.15, '02'],
        '16-20%' : [0.16 , 0.20, '03'],
        '21-25%' : [0.21 , 0.25, '04'],
        '26-30%' : [0.26 , 0.30, '05'],
        '31-35%' : [0.31 , 0.35, '06'],
        '36-40%' : [0.36 , 0.40, '07'],
        '41-45%' : [0.41 , 0.45, '08'],
        '46-50%' : [0.46 , 0.50, '09'],
        '51-55%' : [0.51 , 0.55, '10'],
        '56-60%' : [0.56 , 0.60, '11'],
        '61-65%' : [0.61 , 0.65, '12'],
        '66-70%' : [0.66 , 0.70, '13'],
        '71-75%' : [0.71 , 0.75, '14'],
        '76-80%' : [0.76 , 0.80, '15'],
        '81-85%' : [0.81 , 0.85, '16'],
        '86-90%' : [0.86 , 0.90, '17'],
        '91-95%' : [0.91 , 0.95, '18'],
        '96-100' : [0.96 , 1.00, '19'],
    }
    def __init__(self, value):
        assert isinstance(value, float) or isinstance(value, int), f'value must be a float/int: {type(value)}'
        assert 0.0 <= value <= 1.0, f'invalid range: {value}'
        self.value = float(value)

    def range(self, prefix=False):
        assert isinstance(prefix, bool), f'type mismatch for prefix: {type(prefox)}'
        for key, item in self.AREA_COVERAGE_RANGE_DICT.items():
            if item[0] <= self.value <= item[1]:
                if prefix:
                    return item[2] + ' ' + key
                else:
                    return key
        assert False, f'this must not be reached: self.value = {self.value} ({type(self.value)})'

    def __str__(self):
        return f'{self.value} {self.range()}'
# end-of class AreaCoverage()

# 印刷機
# --------------------------------
class PrintingMachine(): 

    NAMES_DICT = {
        # 印刷機の name は 'PM1' と呼ぶ
        'PM1': {
            # インク消費量原単位
            'INK_CONSUMPTION_PER_M2': 5.5,  # 数字は架空

            # 総インク消費量
            'TOTAL_INK_CONSUMPTION': 100000,  # 数字は架空

            # 用紙サイズ別の印刷枚数 [枚]
            'PAPER_SIZE_DISTRIBUTION_LOCAL': {
                '葉書' :   500,  # 数字は架空
                'A4'   : 60000,
                'B4'   :  1000,
                'A3'   : 900000,
                '長尺' :  10000,
            },

            # ページ長分布 [ページ] : [ジョブ] (両面時は2ページ/枚、片面時は1ページ/枚)
            'PAGE_LENGTH_DISTRIBUTION_LOCAL': {
                '1-10'      : 3000,  # 数字は架空
                '11-20'     : 500,
                '21-30'     : 300,
                '31-50'     : 300,
                '51-100'    : 600,
                '101-200'   : 3000,
                '201-300'   : 700,
                '301-500'   : 500,
                '501-1000'  : 1000,
                '1001-2000' : 500,
                '2001'      : 0,
            },
        },
    }

    if os.path.exists(import_file):
        NAMES_DICT = import_file_var.NAMES_DICT

    NAMES = list(NAMES_DICT.keys())  # ['PM1']

    def __init__(self, name=None):
        assert isinstance(name, str), f'name is required: name={name} ({type(name)})'
        assert name in self.NAMES, f'name must be in {self.NAMES}'
        self.name = name

        self.INK_CONSUMPTION_PER_M2        = self.NAMES_DICT[name]['INK_CONSUMPTION_PER_M2']
        self.TOTAL_INK_CONSUMPTION         = self.NAMES_DICT[name]['TOTAL_INK_CONSUMPTION']
        self.PAPER_SIZE_DISTRIBUTION_LOCAL = self.NAMES_DICT[name]['PAPER_SIZE_DISTRIBUTION_LOCAL']

        self.PAPER_SIZE_DISTRIBUTION_LOCAL_IN_PAGES = []
        for paper_size in Paper().PAPERS:
            self.PAPER_SIZE_DISTRIBUTION_LOCAL_IN_PAGES.append(self.PAPER_SIZE_DISTRIBUTION_LOCAL[paper_size])
        # logger_debug(f'class PrintingMachine: name={name} PAPER_SIZE_DISTRIBUTION_LOCAL_IN_PAGES = {self.PAPER_SIZE_DISTRIBUTION_LOCAL_IN_PAGES}')

        self.PAGE_LENGTH_DISTRIBUTION_LOCAL = self.NAMES_DICT[name]['PAGE_LENGTH_DISTRIBUTION_LOCAL']
        # logger_debug(f'class PrintingMachine: name={name} PAGE_LENGTH_DISTRIBUTION_LOCAL = {self.PAGE_LENGTH_DISTRIBUTION_LOCAL}')

        self.PAGE_LENGTH_RANGE_LIST = []
        for item in list(self.PAGE_LENGTH_DISTRIBUTION_LOCAL.keys()):
            self.PAGE_LENGTH_RANGE_LIST.append(item)
        # logger_debug(f'class PrintingMachine: name={name} PAGE_LENGTH_RANGE_LIST = {self.PAGE_LENGTH_RANGE_LIST}')

        self.PAGE_LENGTH_DISTRIBUTION_LOCAL_IN_PAGES = []
        for item in self.PAGE_LENGTH_RANGE_LIST:
            self.PAGE_LENGTH_DISTRIBUTION_LOCAL_IN_PAGES.append(self.PAGE_LENGTH_DISTRIBUTION_LOCAL[item])
        # logger_debug(f'class PrintingMachine: name={name} PAGE_LENGTH_DISTRIBUTION_LOCAL_IN_PAGES = {self.PAGE_LENGTH_DISTRIBUTION_LOCAL_IN_PAGES}')
    # end-of def __init__

    def __str__(self):
        # return f'{self.name} {self.INK_CONSUMPTION_PER_M2} {self.TOTAL_INK_CONSUMPTION} {self.PAPER_SIZE_DISTRIBUTION_LOCAL} {self.PAGE_LENGTH_DISTRIBUTION_LOCAL}'
        return f'{self.name} {self.INK_CONSUMPTION_PER_M2} {self.TOTAL_INK_CONSUMPTION} 後略'
    # end-of def __str__
# end-of class PrintingMachine()

class PrintingJobResults():
    def __init__(self):
        self.printing_job = []
        self.ink          = []
        self.paper_number = []

    def append(self, printing_job, ink, paper_number):
        assert isinstance(printing_job, PrintedMatter), f'{type(printing_job)}'
        assert isinstance(ink, float), f'{type(ink)}'

        self.printing_job.append(printing_job)
        self.ink.append(ink)
        self.paper_number.append(paper_number)

    def __iter__(self):
        l = []
        for printing_job, ink, paper_number in zip(self.printing_job, self.ink, self.paper_number):
            l.append([printing_job, ink, paper_number])
        return iter(l)

    def __str__(self):
        return [
            f'len={len(self.printing_job)}' + ' ' + 
            f'printing_job[0:4]={self.printing_job[0:4]}' + '\t' + 
            f'ink[0:4]={self.ink[0:4]}' + '\t' + 
            f'paper_number[0:4]={self.paper_number[0:4]}' 
        ][0]

    def total_ink(self):
        return int(sum(self.ink) * 100.0)/100.0

    def len(self):
        return len(self.printing_job)

    def pruning(self):
        '''サイズ縮小'''
        self.printing_job = None
        return self

# end-of class PrintingJobResults()

# 1. 印刷ジョブ実行のシミュレーション
# --------------------------------

# ジョブのインク消費量の計算
def ink_consumption_per_job(printing_machine, printing_job):
    '''ジョブのインク消費量の計算 (page_per_setはページ単位であり、この計算においては両面/片面の考慮は不要)'''
    pages_per_paper_dict = {
        'simplex': 1,
        'duplex' : 2,
    }

    area_coverage          = printing_job.area_coverage
    page_per_set           = printing_job.page_length
    # set_per_job            = printing_job.set_per_job
    duplex_or_simplex      = printing_job.duplex_or_simplex
    paper_size             = printing_job.paper_size
    INK_CONSUMPTION_PER_M2 = printing_machine.INK_CONSUMPTION_PER_M2

    # paper_area = PAGE_AREA_IN_M2[paper_size]
    paper = Paper(paper_size)
    paper_area = paper.page_area                               # [m2/page]

    ink_per_page = paper_area * area_coverage * INK_CONSUMPTION_PER_M2       # [g/page]
    total_ink = ink_per_page * page_per_set                                  # [g/job]
    # total_ink = ink_per_page * page_per_set * set_per_job                    # [g/job]
    total_ink = round(total_ink * 1000.0)/1000.0   # 小数点3桁まで保存

    # 用紙枚数
    paper_number = math.ceil(page_per_set / pages_per_paper_dict[duplex_or_simplex])

    # logger_debug(f'paper_size={paper_size}\tpaper_area={paper_area}page_per_set={page_per_set}\tarea_coverage={area_coverage}\ttotal_ink={total_ink}')

    return total_ink, paper_number
# end-of def ink_consumption_per_job

# assert ink_consumption_per_job('A4', 1, 1.0, INK_CONSUMPTION_PER_M2) == round(0.06237 * 1 * 1.0 * INK_CONSUMPTION_PER_M2 * 1000.0)/1000.0

# 印刷ジョブ実行のシミュレーション
# --------------------------------

def simulate_job_printing(customer, printing_machine):
    total_ink = 0
    printingjob_results = PrintingJobResults()

    # 未知パラメータに基づく印刷ジョブを繰り返して、総インク消費量が目標値に達するまで印刷ジョブを生成する
    while True:
        # 印刷ジョブをランダムに生成
        printing_job = PrintedMatter(customer, printing_machine)

        # ジョブのインク消費量の計算
        ink_consumption, paper_number = ink_consumption_per_job(printing_machine, printing_job)

        # 記録
        printingjob_results.append(printing_job, ink_consumption, paper_number)

        # 総インク消費量
        total_ink = total_ink + ink_consumption

        # 総インク消費量制約に達した時点で (1回の) シミュレーション終了
        if printing_machine.TOTAL_INK_CONSUMPTION <= total_ink:
            break
        continue
    # end-of while True

    return (int(total_ink), printingjob_results)
# end-of def simulate_job_printing

# --------------------------------
# 2. 仮説の妥当性評価
# --------------------------------

def validate_results(customer, printing_machine, total_ink, printingjob_results, mp_savedir_path):
    assert isinstance(customer, Customer), f'{type(customer)}'
    assert isinstance(printingjob_results, PrintingJobResults), f'{type(printingjob_results)}'

    def local_debug_print(s):
        # if args.debug:
        #     logger_debug(s)
        pass

    global cj
    cj = printingjob_results  # デバッグ用のglobal変数

    # モンテカルロ法の総実行回数
    # --------------------------------
    sim_iterations = len(printingjob_results.printing_job)
    local_debug_print(f'    sim_iterations = {sim_iterations}')

    # 評価1: 用紙サイズ別の印刷枚数 [枚]
    # --------------------------------
    number_of_papers_by_paper_size = {}
    for paper_size in Paper().PAPERS:
        number_of_papers_by_paper_size[paper_size] = 0

    for printing_job, ink_consumption, paper_number in printingjob_results:
        number_of_papers_by_paper_size[printing_job.paper_size] += paper_number
    local_debug_print(f'    number_of_papers_by_paper_size={number_of_papers_by_paper_size}')

    p1 = printing_machine.PAPER_SIZE_DISTRIBUTION_LOCAL_IN_PAGES
    q1 = [number_of_papers_by_paper_size[paper_size] for paper_size in Paper().PAPERS]

    local_debug_print(f'      p1 = {p1} [枚] {Paper().PAPERS}')
    local_debug_print(f'      q1 = {q1}')

    h1_ce = cross_entropy(p1, q1)  # , Paper().PAPERS)
    local_debug_print(f'    h1_ce = {h1_ce:.2f} bit')

    # 評価2: 用紙枚数総計
    # --------------------------------
    # logger_debug(f'sum(p1)={sum(p1)}')
    # logger_debug(f'sum(q1)={sum(q1)}')
    sumq1_sump1_ratio = sum(q1) / sum(p1)
    # logger_debug(f'sumq1_sump1_ratio={sumq1_sump1_ratio:.2f}')

    # 評価3: ページ長分布
    # --------------------------------
    number_of_jobs_by_page_length = {}
    for page_length in printing_machine.PAGE_LENGTH_RANGE_LIST:
        number_of_jobs_by_page_length[page_length] = 0

    for printing_job, ink_consumption, paper_number in printingjob_results:
        pl_range = page_length_range_str(printing_job.page_length)   # 1 -> '1-10'
        number_of_jobs_by_page_length[pl_range] += 1
    local_debug_print(f'    number_of_jobs_by_page_length = {number_of_jobs_by_page_length}')

    p2 = printing_machine.PAGE_LENGTH_DISTRIBUTION_LOCAL_IN_PAGES
    q2 = [number_of_jobs_by_page_length[page_length] for page_length in printing_machine.PAGE_LENGTH_RANGE_LIST]

    local_debug_print(f'      p2 = {p2} [ジョブ/ページ数] ')
    local_debug_print(f'      q2 = {q2}')

    h2_ce = cross_entropy(p2, q2)  # printing_machine.PAGE_LENGTH_RANGE_LIST)
    local_debug_print(f'    h2_ce = {h2_ce:.2f} bit')

    # --------------------------------
    # Excelファイルとして保存
    # --------------------------------
    def save_to_excel():
        columns = ['matter', 'area_coverage', 'duplex_or_simplex', 'paper_size', 'page_length', 'paper_number', 'ink_consuption', 'pl_range', 'ac_range', 'ink_range', 'ps_range',]
        data = []

        for printing_job, ink_consumption, paper_number in printingjob_results:
            pl_range = page_length_range_str(printing_job.page_length, prefix=True)   # 1 -> '1-10'
            ac_range = area_coverage_range_str(printing_job.area_coverage, prefix=True)
            ink_range = int( ink_consumption / 100.0 ) * 100
            ps_range = page_size_str(printing_job.paper_size, prefix=True)

            data.append((
                printing_job.type,
                printing_job.area_coverage,
                printing_job.duplex_or_simplex,
                printing_job.paper_size,
                printing_job.page_length,
                # printing_job.set_per_job,
                paper_number,
                ink_consumption,
                pl_range,
                # set_per_job_range,
                ac_range,
                ink_range,
                ps_range,
            ))

        df = pd.DataFrame(data=data, columns=columns)
        df.name = 'df'
        df.index.name = 'index'
        
        filename = f'{mp_savedir_path}/{printing_machine.name} {customer.customer_type}-{customer.area_coverage_lvl}-{customer.page_length_lvl} 枚比={sumq1_sump1_ratio:.2f} h1_ce={h1_ce:.2f} h2_ce={h2_ce:.2f} {ok_or_ng}{"★" if ok_or_ng == "OK" else ""}.xlsx'
        df.to_excel(filename)
    # end-of def save_to_excel

    # --------------------------------
    # グラフを作成して保存
    # --------------------------------
    def save_to_chart(showfig=False, savefig=False):

        # --------------------------------
        # サポート関数
        # --------------------------------

        # (サポート1) 用紙サイズ別のジョブ数 (グラフ用)
        # --------------------------------
        def get_number_of_jobs_by_paper_size():
            number_of_jobs_by_paper_size = {}
            for paper_size in Paper().PAPERS:
                number_of_jobs_by_paper_size[paper_size] = 0

            for printing_job, ink_consumption, paper_number in printingjob_results:
                number_of_jobs_by_paper_size[printing_job.paper_size] += 1  # ジョブ数をカウント
            local_debug_print(f'    number_of_jobs_by_paper_size={number_of_jobs_by_paper_size}')
            return number_of_jobs_by_paper_size

        # (サポート2) 用紙サイズ別、エリアカバレッジ別のカウント [ジョブ] (グラフ用)
        # --------------------------------
        def get_papersize_areacoverage_papers():
            papersize_areacoverage_papers = {}
            for paper_size in Paper().PAPERS:
                papersize_areacoverage_papers[paper_size] = {}
                for ac_range in list(AreaCoverage.AREA_COVERAGE_RANGE_DICT.keys()):
                    papersize_areacoverage_papers[paper_size][ac_range] = 0

            for printing_job, ink_consumption, paper_number in printingjob_results:
                paper_size = printing_job.paper_size
                ac_range = AreaCoverage(printing_job.area_coverage).range()
                # logger_debug(f'paper_size={paper_size}\tac_range={ac_range}')
                papersize_areacoverage_papers[paper_size][ac_range] += 1
            # logger_debug(f'papersize_areacoverage_papers={papersize_areacoverage_papers}')
            return papersize_areacoverage_papers

        # (サポート3 )用紙サイズ別、印刷ページ長別のカウント [ジョブ] (グラフ用)
        # --------------------------------
        def get_papersize_pagelength_jobs():
            papersize_pagelength_jobs = {}
            for paper_size in Paper().PAPERS:
                papersize_pagelength_jobs[paper_size] = {}
                for page_length in printing_machine.PAGE_LENGTH_RANGE_LIST:
                    papersize_pagelength_jobs[paper_size][page_length] = 0

            for printing_job, ink_consumption, paper_number in printingjob_results:
                pl_range = page_length_range_str(printing_job.page_length)   # 1 -> '1-10'
                paper_size = printing_job.paper_size
                papersize_pagelength_jobs[paper_size][pl_range] += 1
            # logger_debug(f'papersize_pagelength_jobs={papersize_pagelength_jobs}')
            return papersize_pagelength_jobs

        # (サポート4) 用紙サイズ別、印刷ページ長別のカウント [枚] (グラフ用)
        # --------------------------------
        def get_papersize_pagelength_papers():
            papersize_pagelength_papers = {}
            for paper_size in Paper().PAPERS:
                papersize_pagelength_papers[paper_size] = {}
                for page_length in printing_machine.PAGE_LENGTH_RANGE_LIST:
                    papersize_pagelength_papers[paper_size][page_length] = 0

            for printing_job, ink_consumption, paper_number in printingjob_results:
                paper_size = printing_job.paper_size
                pl_range = page_length_range_str(printing_job.page_length)   # 1 -> '1-10'
                # logger_debug(f'paper_size={paper_size}\tpl_range={pl_range}\tpaper_number={paper_number}')
                papersize_pagelength_papers[paper_size][pl_range] += paper_number
            # logger_debug(f'papersize_pagelength_papers={papersize_pagelength_papers}')
            return papersize_pagelength_papers

        # --------------------------------
        # プロット作成
        # --------------------------------

        # プロット用の情報を収集
        ok_or_ng = ok_or_ng_decision(h1_ce, h2_ce, sumq1_sump1_ratio)

        events1 = Paper().PAPERS
        events2 = printing_machine.PAGE_LENGTH_RANGE_LIST

        # 新しい図を作成
        fig = init_figure()

        gs = gridspec.GridSpec(7, 6)

        # --------------------------------
        # |  グラフ1  |  グラフ2         |
        # --------------------------------
        #             |  グラフ3         |
        #             --------------------
        #             |  グラフ4         |
        #             --------------------
        #             |  グラフ5         |
        # --------------------------------
        # |  グラフ6  |  グラフ7         |
        # --------------------------------
        # |          グラフ8             |
        # --------------------------------
        # |          グラフ9             |
        # --------------------------------

        # --------------------------------
        # グラフ1. 印刷機の使われ方 (推定)
        # --------------------------------
        def plot_for_machine_usage():
            ac_ranges = {'H':0,'M':1,'L':2}
            pl_ranges = {'L':0,'M':1,'H':2}
            market = np.array([[0,0,0],[0,0,0],[0,0,0]])  # 1行目, 2行目, 3行目
            market[ac_ranges[customer.area_coverage_lvl]][pl_ranges[customer.page_length_lvl]] = 1
            cmap = mcolors.ListedColormap(['w', 'tab:pink'])

            ax = fig.add_subplot(gs[0, 0])
            im = ax.imshow(market, cmap=cmap)
            ax.set_xticks(range(len(pl_ranges)), pl_ranges)
            ax.set_yticks(range(len(ac_ranges)), ac_ranges)

            q = 1
            for i in reversed(range(len(ac_ranges))):
                for j in range(len(pl_ranges)):
                    text = ax.text(j, i, f'Q{q}', ha='center', va='center', color='k')
                    q += 1

            ax.set_title('印刷機の使われ方 (推定)')
            ax.set_xlabel('印刷ジョブ長')
            ax.set_ylabel('エリアカバレッジ')

        # end-of def plot_for_machine_usage()
        plot_for_machine_usage()


        # --------------------------------
        # グラフ2. ヒストグラム - エリアカバレッジ
        # --------------------------------
        pm = PrintedMatter(None, None)  # このインスタンスは、インスタンス変数の参照用として用いる

        papersize_areacoverage_counts = {}
        for paper_size in Paper().PAPERS:
            papersize_areacoverage_counts[paper_size] = []
        for printing_job, ink_consumption, paper_number in printingjob_results:
            paper_size = printing_job.paper_size
            area_coverage = printing_job.area_coverage
            papersize_areacoverage_counts[paper_size].append(area_coverage)

        for i, paper_size in enumerate([item for item in list(pm.area_coverage_list.keys()) if item in ['葉書','A4','B4','A3','長尺']]):
            if 0 < sum(papersize_areacoverage_counts[paper_size]):
                # logger_debug(f'sum(papersize_areacoverage_counts[paper_size]) = {sum(papersize_areacoverage_counts[paper_size])}')
                ax = fig.add_subplot(gs[0, i+1])
                sns.histplot(papersize_areacoverage_counts[paper_size], bins=50, binrange=[0.0, 1.0],
                             edgecolor='r', linewidth=0, color='tab:pink',)
                plt.title(paper_size)

        # --------------------------------
        # グラフ3. ヒストグラム - 印刷ページ長
        # --------------------------------
        papersize_pagelength_counts = {}
        for paper_size in Paper().PAPERS:
            papersize_pagelength_counts[paper_size] = []
        for printing_job, ink_consumption, paper_number in printingjob_results:
            paper_size = printing_job.paper_size
            page_length = printing_job.page_length
            papersize_pagelength_counts[paper_size].append(page_length)

        for i, paper_size in enumerate([item for item in list(pm.page_length_list.keys()) if item in ['葉書','A4','B4','A3','長尺']]):
            if 0 < sum(papersize_pagelength_counts[paper_size]):
                ax = fig.add_subplot(gs[1, i+1])
                sns.histplot(papersize_pagelength_counts[paper_size], bins=50, binrange=[0,1000],
                             edgecolor='r', linewidth=0, color='tab:pink',)
                # plt.title(paper_size)  # 冗長なので非表示

        # --------------------------------
        # グラフ4. 用紙サイズの比率 (推定)
        # --------------------------------
        matter_dic = {}
        pm = PrintedMatter(None, None).PRINTED_MATTERS  # このインスタンスは、インスタンス変数の参照用として用いる

        for matter in pm.keys():
            paper_size_list_str = matter
            if matter in customer.matters.keys():
                matter_dic[paper_size_list_str] = customer.matters[matter]
            else:
                matter_dic[paper_size_list_str] = 0

        ax = fig.add_subplot(gs[2,1:])

        plt.bar(matter_dic.keys(), matter_dic.values(), color='tab:pink')
        plt.title('  用紙サイズ別の印刷ジョブ数 (推定)', loc='left', y=1.0, pad=-14)
        plt.ylabel('ジョブ数比 (%)')

        # --------------------------------
        # グラフ5. 推定したジョブ数
        # --------------------------------
        # plt.subplot(6,1,3)
        ax = fig.add_subplot(gs[3,1:])

        ps_ranges = list(get_number_of_jobs_by_paper_size().keys())
        ind = np.arange(len(ps_ranges))
        width = 0.4
        
        # バー左側: 推定(エリアカバレッジ)
        # --------------------------------
        df = pd.DataFrame(get_papersize_areacoverage_papers())
        # logger_debug(f'df (推定(エリアカバレッジ) [ジョブ]) =\n{df}')

        # 各用紙サイズのジョブ数の和がゼロの行を捨てる
        df = df.loc[ 0 < df.apply(np.sum, axis=1) ].copy()
        # logger_debug(f'df (左: 推定(エリアカバレッジ) [ジョブ]) (各用紙サイズのジョブ数の和がゼロの行を捨てる) =\n{df}')

        bottom = np.zeros(len(ps_ranges))
        for ac_range, rest in df.iterrows():
            plt.bar(ind, rest, width, label=ac_range, bottom=bottom)
            bottom += rest
        # plt.legend(loc='upper left') # 凡例は表示しない (左右の柱で異なるが両立できなかった)
            
        # バー右側: 推定(印刷ページ長)
        # --------------------------------
        df = pd.DataFrame(get_papersize_pagelength_jobs())
        # logger_debug(f'df (右: 推定(印刷ページ長) [ジョブ]) =\n{df}')

        bottom = np.zeros(len(ps_ranges))
        for paper_size, rest in df.iterrows():
            plt.bar(ind + width, rest, width, label=paper_size, bottom=bottom)
            bottom += rest
        
        # plt.xlabel('用紙サイズ')
        plt.ylabel('ジョブ')
        plt.title('  推定したジョブ数 (左=エリアカバレッジ、右=印刷ページ長、色は区別のため)', loc='left', y=1.0, pad=-14)
        
        # xticks()
        # First argument - A list of positions at which ticks should be placed
        # Second argument -  A list of labels to place at the given locations
        plt.xticks(ind + width / 2, ps_ranges)
        
        # Finding the best position for legends and putting it
        # plt.legend(loc='upper right') # 凡例は表示しない (左右の柱で異なるが両立できなかった)
        del ps_ranges, ind, width, df, bottom, ac_range, paper_size, rest

        # --------------------------------
        # グラフ6. 総インク消費量
        # --------------------------------
        def plot_for_total_ink():
            expected_total_ink_consumption_in_k  = int(printing_machine.TOTAL_INK_CONSUMPTION/1000)
            predicted_total_ink_consumption_in_k = int(total_ink/1000)

            ax = fig.add_subplot(gs[4, 0])
            df = pd.DataFrame(data={'期待': [expected_total_ink_consumption_in_k], '推定': [predicted_total_ink_consumption_in_k]})
            ax = sns.barplot(data=df, palette=['tab:gray', 'tab:pink'], legend=False, estimator="sum", errorbar=None)
            ax.set_ylabel('インク消費量 [kg]')

            # グラフの幅を縮小する
            ll, bb, ww, hh = ax.get_position().bounds
            # print((ll, bb, ww, hh))
            ax.set_position([ll, bb, ww*0.75, hh]) 

            plt.title(f'  推定したインク消費量\n\n  ・推定={expected_total_ink_consumption_in_k:,} kg\n  ・期待={predicted_total_ink_consumption_in_k:,} kg\n', loc='left', y=+1.1, pad=-14)   # タイトルを上げる: yを増やす、タイトルは下がる: yを減らす

            # plt.subplots_adjust(left=-0.05, right=1.05)

        # end-of def plot_for_total_ink

        plot_for_total_ink()


        # --------------------------------
        # グラフ7. 推定した印刷枚数
        # --------------------------------
        # plt.subplot(6,1,4)
        ax = fig.add_subplot(gs[4,1:])

        ps_ranges = events1
        ind = np.arange(len(ps_ranges))
        width = 0.4

        # バー左側: 期待
        # --------------------------------
        df_hidden = pd.DataFrame(data=[p1], columns=ps_ranges)
        # logger_debug(f'df_hidden (期待 [枚]) =\n{df_hidden}')
        plt.bar(ind, df_hidden.iloc[0].values.tolist(), width, label='期待', color='tab:gray')
        
        # バー右側: 推定
        # --------------------------------
        global df_debug
        df = pd.DataFrame(get_papersize_pagelength_papers())
        df_debug = df.copy()
        # logger_debug(f'df (推定 [枚]) =\n{df}')
        bottom = np.zeros(len(ps_ranges))
        for paper_size, rest in df.iterrows():
            plt.bar(ind + width, rest, width, label=paper_size, bottom=bottom)
            bottom += rest
        
        # plt.xlabel('用紙サイズ')
        plt.ylabel('印刷枚数 [枚]')
        plt.title(f'  推定した印刷枚数\n\n  ・推定={int(df.sum().sum()/1000):,} k枚\n  ・期待={int(sum(p1)/1000):,} k枚\n  ・比率 (推定/期待)={(df.sum().sum() / sum(p1)):0.2f}', loc='left', y=+0.5, pad=-14)   # タイトルを上げる: yを増やす、タイトルは下がる: yを減らす
        
        # xticks()
        # First argument - A list of positions at which ticks should be placed
        # Second argument -  A list of labels to place at the given locations
        plt.xticks(ind + width / 2, ps_ranges)
        
        # Finding the best position for legends and putting it
        plt.legend(loc='upper right', prop={'size': 6})
        # plt.show()
        del ps_ranges, ind, width, df_hidden, df, bottom, paper_size, rest

        # --------------------------------
        # グラフ8. 期待される印刷ページ長の分布
        # --------------------------------
        # plt.subplot(6,1,5)
        ax = fig.add_subplot(gs[5,:])
        plt.bar(events2, p2, color='tab:gray')
        plt.title('  期待される印刷ページ長の分布', loc='left', y=1.0, pad=-14)
        plt.ylabel('ジョブ')

        # --------------------------------
        # グラフ9. 推定した印刷ページ長の分布
        # --------------------------------
        # plt.subplot(6,1,6)
        ax = fig.add_subplot(gs[6,:])

        df = pd.DataFrame(get_papersize_pagelength_jobs())
        # logger_debug(f'df (推定 [ジョブ])=\n{df}')
        # logger_debug(f'df.T (推定 [ジョブ])=\n{df.T}')
        df = df.T
        width = 0.8
        pl_ranges = df.columns.tolist()
        bottom = np.zeros(len(pl_ranges))
        for paper_size, rest in df.iterrows():
            plt.bar(pl_ranges, rest, width, label=paper_size, bottom=bottom)
            bottom += rest
        plt.title('  推定した印刷ページ長の分布', loc='left', y=1.0, pad=-14)
        plt.ylabel('ジョブ')
        plt.legend(loc='upper right')
        del df, width, pl_ranges, bottom, paper_size, rest

        # --------------------------------
        # グラフ全体のタイトル
        # --------------------------------
        plt.suptitle(f'name={printing_machine.name} [TYPE={customer.customer_type} AC={customer.area_coverage_lvl} PL={customer.page_length_lvl}] Matters={customer.matters} ⇒ 推定[枚]/期待[枚]={sumq1_sump1_ratio:.2f} h1_ce={h1_ce:.2f} bit h2_ce={h2_ce:.2f} bit {"★" if ok_or_ng == "OK" else ""}' )

        # --------------------------------
        # グラフを画像として保存
        # --------------------------------
        if savefig:
            filename = f'{mp_savedir_path}/{printing_machine.name} {customer.customer_type}-{customer.area_coverage_lvl}-{customer.page_length_lvl} 枚比={sumq1_sump1_ratio:.2f} h1_ce={h1_ce:.2f} h2_ce={h2_ce:.2f} {ok_or_ng}{"★" if ok_or_ng == "OK" else ""}.png'
            plt.savefig(filename)

        if showfig:
            plt.show()
    # end-of def save_to_chart

    # --------------------------------
    # OK-NG 判定
    # --------------------------------
    def ok_or_ng_decision(h1_ce, h2_ce, sumq1_sump1_ratio):

        # 閾値は経験的に決めた
        h1_ce_lim = customer.h1_ce_lim
        h2_ce_lim = customer.h2_ce_lim

        sumq1_sump1_ratio_llim = 0.70
        sumq1_sump1_ratio_ulim = 1.30
        if (h1_ce <= h1_ce_lim) and (h2_ce <= h2_ce_lim) and (sumq1_sump1_ratio_llim <= sumq1_sump1_ratio <= sumq1_sump1_ratio_ulim):
            return 'OK'
        else:
            return 'NG'
    # end-of ok_or_ng_decision

    ok_or_ng = ok_or_ng_decision(h1_ce, h2_ce, sumq1_sump1_ratio)

    if ok_or_ng == 'OK':
        # 比較用グラフ保存
        save_to_chart(showfig=False, savefig=True)

        # Excelファイルとして保存 (かなり待ち時間がかかるので、OKの場合のみ保存することとした)
        save_to_excel()
    else:
        # 比較用グラフ保存
        save_to_chart(showfig=False, savefig=True)

        # Excelファイルとして保存 (かなり待ち時間がかかる)
        # save_to_excel()

    result_cross_entropy = {
        'h1_ce': h1_ce, 'h1_p': p1, 'h1_q': q1,
        'h2_ce': h2_ce, 'h2_p': p2, 'h2_q': q2,
    }

    return (ok_or_ng, result_cross_entropy, sim_iterations, sumq1_sump1_ratio)
# end-of def validate_results

# --------------------------------
# モンテカルロ法による印刷シミュレーション
# --------------------------------

def printing_simulation(mp_args):
    '''モンテカルロ法による印刷シミュレーション'''
    name, iterations, mp_savedir_path = mp_args

    # 格納先
    os.makedirs(mp_savedir_path)

    sim_result_for_machine = []     # 基準に適合した「もっともらしい」シミュレーション結果のlist。当該の印刷機のものを格納する。
    num_ok = 0

    # ターゲット印刷機
    printing_machine = PrintingMachine(name)

    # シミュレーションを指定した回数だけ繰り返す。モンテカルロ法として妥当な解を探索する。
    for i in range(iterations):
        # 顧客の印刷機に固有のオンデマンド印刷物の特徴を作成。これを元に印刷ジョブが作成される。
        customer = Customer()

        # 1. 印刷ジョブ実行のシミュレーション
        (total_ink, printingjob_results) = simulate_job_printing(customer, printing_machine)

        # 2. 仮説の妥当性評価
        (   ok_or_ng, result_cross_entropy, sim_iterations, sumq1_sump1_ratio
        ) = validate_results(customer, printing_machine, total_ink, printingjob_results, mp_savedir_path)

        # 3. サマリー
        if ok_or_ng == 'OK':
            num_ok += 1

        logger_debug('  ' + 
            f'MC法 name={name} try={i}' + '\t' +
            f'{str(customer)}' + '  ' +
            f'loop={sim_iterations:>6}' + '  ' +
            f'ink={int(total_ink/1000.0)}' + '  ' +
            f'枚数比={sumq1_sump1_ratio:.2f}' + '  ' +
            f'ce1={result_cross_entropy["h1_ce"]:5.2f}' + '  ' +
            f'ce2={result_cross_entropy["h2_ce"]:5.2f}' +  '  ' +
            f' → {ok_or_ng} (総OK:{num_ok})' + f'{"★" if ok_or_ng == "OK" else ""}'
        )

        # シミュレーション結果の保存。ただしOKの場合のみ、詳細な結果を保存する。消費するメモリサイズを抑制したい。
        if ok_or_ng == 'NG':
            printingjob_results = printingjob_results.pruning()  # サイズ縮小

        sim_result_for_machine.append([
            printing_machine,
            customer,
            printingjob_results,
            ok_or_ng,
            result_cross_entropy,
            sim_iterations,
            total_ink,
        ])
    # end-of for i in range(iterations)

    # h1_ce と h2_ce の最小値をそれぞれ表示する
    def debug_show_min_h1_and_h2(sim_result_for_machine):
        # logger_debug(f'sim_result_for_machine = {sim_result_for_machine}')
        h1_ce_min = None
        h2_ce_min = None
        for item in sim_result_for_machine:
            result_cross_entropy = item[4]
            assert isinstance(result_cross_entropy, dict), f'result_cross_entropy must be a dict: {type(result_cross_entropy)}'
            if h1_ce_min == None:
                h1_ce_min = result_cross_entropy['h1_ce']
            else:
                if result_cross_entropy['h1_ce'] < h1_ce_min:
                    h1_ce_min = result_cross_entropy['h1_ce']

            if h2_ce_min == None:
                h2_ce_min = result_cross_entropy['h2_ce']
            else:
                if result_cross_entropy['h2_ce'] < h2_ce_min:
                    h2_ce_min = result_cross_entropy['h2_ce']
        # end-of for item in sim_result_for_machine

        if not isinstance(h1_ce_min, type(None)):
            # logger_debug(f'h1_ce_min = {h1_ce_min:.2f}')
            pass
        if not isinstance(h2_ce_min, type(None)):
            # logger_debug(f'h2_ce_min = {h2_ce_min:.2f}')
            pass
    # end-of def debug_show_min_h1_and_h2()

    debug_show_min_h1_and_h2(sim_result_for_machine)

    # シミュレーション結果を pickle ファイルとして保存する。そのパス名を返す。
    # 保存する利点
    # (1) シミュレーション結果を詳しく分析したいときに、再現させるために役に立つ
    # (2) コンピュータのメモリをより効率的に使用する
    pickle_file = os.path.join(mp_savedir_path, f'{name}.pickle.gz')
    logger_debug(f'シミュレーション結果を、pickle ファイルへ保存する(BGN): {pickle_file}')
    with gzip.open(pickle_file, 'wb') as f:
        pickle.dump(sim_result_for_machine, f)
    logger_debug(f'シミュレーション結果を、pickle ファイルへ保存する(END)')

    return pickle_file  # pickle ファイルのパス名を返す
# end-of def printing_simulation(name)

def generate_monte_carlo_simulation(iterations):
    '''モンテカルロ法による印刷シミュレーション'''

    # シミュレーション対象の印刷機を列挙する。リスト names にシミュレーションで使用する印刷機の名前を格納。
    names = PrintingMachine.NAMES   # ['No1', 'No2', ..]
    logger.debug(f'name = {names}')
    if 1 <= len(args.printing_machines):
        for name in args.printing_machines:
            assert name in names, f'no name found: {name}\navailable names: {names}'
        names = [name for name in names if name in args.printing_machines]
    logger.debug(f'次の印刷機について順次シミュレーションを行う: names={names}')

    if single_processing():
        # シミュレーションを直列で行う (--cpu_count がない場合、これがデフォルト)
        # 利点
        #   シミュレーション時の画面出力がログファイルに記録される。デバッグ時に役立つ。
        #   シミュレーション時の詳細なデータが pickle ファイルに記録される。デバッグ時に役立つ。
        mp_results = []
        for name in names:
            mp_args = [name, iterations, os.path.join(savedir_path, name)]
            # logger.debug(f'mp_args={mp_args}')
            results = printing_simulation(mp_args)
            # logger.debug(f'results={results}')
            mp_results.append(results)
        # logger.debug(f'mp_results={mp_results}')
    else:
        # シミュレーションを並列で行う (--cpu_count で 2 以上を指定した場合)
        # 利点
        #   シミュレーション時間がより短かくなる
        # 未解決の問題
        #   (1) multiprocessing の場合、親プロセスの logger を子プロセスが取得できなかった(継承されなかった)。
        #       --> multiprocessing.get_logger() を使う必要があると思う。
        #           https://docs.python.org/ja/3.13/library/multiprocessing.html#logging
        #   (2) multiprocessing の場合、
        #       子プロセスで作成して返したオブジェクトを、親プロセスで pickle ファイルとして保存できなかった。
        #       理由: 親プロセスの class と、子プロセスの class は別ものであることで、正常な動作として説明できる。
        mp_args = []
        for name in names:
            mp_args.append([name, iterations, os.path.join(savedir_path, name)])
        # logger.debug(f'mp_args={mp_args}')

        pool = mp.Pool(
            processes = args.cpu_count,  # CPU
            maxtasksperchild = 1,        # ワーカープロセスは、1つのタスクを終えた時点で exit する。
        )
        mp_results = pool.map(printing_simulation, mp_args)
        pool.close()
        pool.join()
        # logger.debug(f'mp_results={mp_results}')
    # end-of if single_processing()

    # シミュレーション結果を pickle ファイルから再生し、一つにまとめる
    sim_result_all = []     # 基準に適合した「もっともらしい」シミュレーション結果のlist。全ての印刷機の合計を格納する。
    for name in names:
        # 子プロセスで作成して返したオブジェクトは pickle ファイルとして保存されている
        pickle_file = os.path.join(os.path.join(savedir_path, name), f'{name}.pickle.gz')
        with gzip.open(pickle_file, 'rb') as f:
            sim_result_for_machine = pickle.load(f)
        sim_result_all.extend(sim_result_for_machine)
    # logger.debug(f'len(sim_result_all) = {len(sim_result_all)}')

    if single_processing():
        # シングルプロセスの場合のみ、一つにまとめたシミュレーション結果を pickle ファイルとして保存する。
        # このコードは直列処理をした場合のみ機能する。
        pickle_file = distination_pathname(dt=False, filename=f'sim_result.pickle.gz')
        logger.debug(f'一つにまとめたシミュレーション結果を、pickle ファイルへ保存する(BGN): {pickle_file}')
        with gzip.open(pickle_file, 'wb') as f:
            pickle.dump(sim_result_all, f)
        logger.debug(f'一つにまとめたシミュレーション結果を、pickle ファイルへ保存する(END)')
    # end-of if single_processing()

    return sim_result_all
# end-of def generate_monte_carlo_simulation(iterations)

# --------------------------------
# シミュレーション結果の表示
# --------------------------------

def show_results(sim_result_all):
    '''シミュレーション結果の表示'''

    # 新しい図を作成
    # --------------------------------
    fig = init_figure()

    subplot_id     = 1
    subplot_width  = 12
    subplot_height = math.ceil( (12 * len(sim_result_all)) / subplot_width )  # math.ceil() で切り上げ
    max_plots      = 10  # 表示するシミュレーション結果の最大数
    axes={}

    # サブプロット作成
    # --------------------------------
    def plot_for_machine_usage():
        subplot_id = 511

        ac_ranges = {'H':0,'M':1,'L':2}
        pl_ranges = {'L':0,'M':1,'H':2}

        fig = init_figure()

        global ok_results
        ok_results = []
        for i, rest in enumerate(sim_result_all):
            printing_machine = rest[0]
            customer = rest[1]
            ok_or_ng = rest[3]

            ok_results.append([
                printing_machine.name,
                customer.area_coverage_lvl,
                customer.page_length_lvl,
                ok_or_ng
            ])
        global df
        df = pd.DataFrame(ok_results, columns = ['name','area_coverage_lvl','page_length_lvl','ok_or_ng'])

        global market
        market = []
        ok_df = df.loc[df['ok_or_ng']=='OK']
        print(f'ok_df={ok_df}')
        if 1 <= len(ok_df):
            for name, rest in ok_df.groupby(by='name'):
                print(f'name = {name} rest = \n{rest}')

                market = np.array([[0,0,0],[0,0,0],[0,0,0]])  # 1行目, 2行目, 3行目
                for j, each_ok in rest.iterrows():
                    market[ac_ranges[each_ok.area_coverage_lvl]][pl_ranges[each_ok.page_length_lvl]] += 1

                ax = fig.add_subplot(subplot_id)
                subplot_id += 1

                im = ax.imshow(market)  # ヒートマップ
                ax.set_xticks(range(len(pl_ranges)), pl_ranges)
                ax.set_yticks(range(len(ac_ranges)), ac_ranges)

                q = 1
                for i in reversed(range(len(ac_ranges))):
                    for j in range(len(pl_ranges)):
                        text = ax.text(j, i, market[i][j], ha='center', va='center', color='k')
                        q += 1
                ax.set_title(f'印刷機: {name}', loc='left')
                ax.set_xlabel('印刷ジョブ長')
                ax.set_ylabel('エリアカバレッジ')

            plt.suptitle('印刷機毎の使われ方')
            fig.tight_layout()

            # if args.show:
            #     plt.show()
            # else:
            if True:
                filename = f'印刷機毎の使われ方.png'
                plt.savefig(distination_pathname(dt=True, filename=filename))
                logger.debug(f'filename = {filename}')

        else:
            logger.debug(f'表示するシミュレーション結果はなかったのでグラフは表示しない。(len(ok_df)={len(ok_df)})')
            pass
        # end-of if 1 <= len(ok_df)
    # end-of def plot_for_machine_usage()
    plot_for_machine_usage()
# end-of def show_results(sim_result_all)

def show_hidden_parameter(sim_result):
    '''パラメータ推定結果の表示 ( 未完成 )'''
    global area_coverage_parameter_dic
    global feature_parameter

    # 初期化
    area_coverage_parameter_dic={}
    for content_type in content_type_list:
        area_coverage_parameter_dic[content_type] = []
    
    for i, (result, feature_parameter) in enumerate(zip(sim_result, sim_feature_parameter)):
        if result == 'NG':
            continue
        logger.debug(f'{i}\t{result}\t{feature_parameter}')
        # logger.debug(f'{i}\t{rest} ({type(rest)})')
        area_coverage_parameter = feature_parameter[0][1]
        logger.debug(f'{i}\tarea_coverage_parameter = {area_coverage_parameter}')

        for content_type in content_type_list:
            logger.debug(f'area_coverage_parameter[{content_type}] = {area_coverage_parameter[content_type]}')
            area_coverage_parameter_dic[content_type].append(area_coverage_parameter[content_type])

    subplot_id     = 1
    subplot_width  = 3
    subplot_height = 1
    for content_type in content_type_list:
        logger.debug(f'area_coverage_parameter_dic[{content_type}] = {area_coverage_parameter_dic[content_type]}')
        plt.subplot(subplot_height, subplot_width, subplot_id)
        histogram(area_coverage_parameter_dic[content_type])
        subplot_id += 1
    plt.show()
# end-of def show_hidden_parameter(sim_feature_parameter)

# --------------------------------
# メインルーチン
# --------------------------------
def main():
    # global args
    # args = arg_parse()

    # 本スクリプトを保存 (再現性を高める)
    for file in [this_file, import_file]:
        if os.path.exists(file):
            shutil.copyfile(file, distination_pathname(dt=True, filename=file))

    init_logging(logfile=distination_pathname(dt=True, filename='_debug.log'))
    logger.debug('start')
    logger.debug(args)  # コマンドライン引数を記録

    if 0 == args.seed:
        random.seed(None)
    else:
        random.seed(args.seed)

    logger.debug(f'args.seed = {args.seed}')
    logger.debug(f'args.import_file = {args.import_file} ' +  (' 存在する' if os.path.exists(import_file) else '存在しない'))

    # シミュレーション実行する。シミュレーションの結果を pickle ファイルに保存する。
    # ただし、「--pickle パス名」オプションが指定された場合、シミュレーションは行わなず、
    #     pickle ファイルを読み込んで大域変数 sim_result_all に格納し、シミュレーション結果を表示する。
    #     これを利用すると、「シミュレーション結果の表示」の開発に注力することができる。
    global sim_result_all
    if args.pickle:
        pickle_file = args.pickle

        if not os.path.exists(pickle_file):
            logger.debug(f'--pickle オプションで指定されたファイルは見つからなかった: pickle_file={pickle_file}')
            sys.exit(1)
        else:
            # MC法の結果を pickle ファイルからロード
            logger.debug(f'シミュレーション結果ファイルから読み込む (シミュレーションは行わない): {pickle_file}')
            def load_sim_result_all(pickle_file):
                with gzip.open(pickle_file, 'rb') as f:
                    sim_result_all = pickle.load(f)
                return sim_result_all
            sim_result_all = load_sim_result_all(pickle_file)
    else:
        logger.debug(f'シミュレーションを行う。その結果を pickle ファイルへ保存する。')
        # モンテカルロ法
        sim_result_all = generate_monte_carlo_simulation(iterations = args.iterations)
    # end-of if os.path.exists(pickle_file)

    logger.debug(f'シミュレーション結果は global sim_result_all に格納した len(sim_result_all) = {len(sim_result_all)}')

    # シミュレーション結果の表示
    logger.debug(f'シミュレーション結果を表示する')
    show_results(sim_result_all)

    # パラメータ推定結果の表示 ( 未完成 )
    # logger.debug(f'パラメータ推定結果を表示する')
    # show_hidden_parameter(sim_result_all)

    logger.debug('successfully completed')
    logging.shutdown()
# end-of def main

if __name__ == '__main__':
    # __spec__ = None は、次の不具合回避のため
    # pdb & multiprocessing.Pool: AttributeError: module '__main__' has no attribute '__spec__' #87115
    __spec__ = None
    main()
