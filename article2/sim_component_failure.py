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
python sim_component_failure.py --designed_life 1000000 --wearout_rate 1.5 --maxt 60*24*30*12 --maxx 100

■正規分布
from reliability.Distributions import Normal_Distribution
nd = Normal_Distribution(mu=1000000, sigma=10000)
nd.plot()
nd.random_samples(10)

■ワイブル分布
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
wd = Weibull_Distribution(alpha=1000000, beta=2.0)
# wd.plot()
data = wd.random_samples(10)
fit = Fit_Weibull_2P(failures=data,show_probability_plot=False,print_results=False)
X_lower,X_point,X_upper = fit.distribution.CDF(CI_type='time',CI_y=0.7)
plt.show()

'''
import sys
import simpy
import random
import statistics
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import japanize_matplotlib
import argparse
import pandas as pd
import seaborn as sns
# import seaborn.objects as so
from addict import Dict
import math

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P, Fit_Normal_2P

pd.options.display.float_format = '{:.1f}'.format

wait_times = None             # print_job 毎の印刷所要時間
printing_jobs_log = None      # print_job 毎の終了時刻と成否
replacement_parts_log = None  # 交換した部品 [交換理由, 停止時間, 部品情報]

def arg_parse():
    global params

    parser = argparse.ArgumentParser()
    parser.add_argument('--step'          , action='store_true', default=False)
    parser.add_argument('--debug'         , action='store_true', default=False)
    parser.add_argument('--wearout_rate'  , type=float , nargs='+', default=[1.0], help='予防保守の管理目標(係数)。部品ライフ設計値を1.0とした場合の管理目標(係数)を指定する。(デフォルト: 1.0)。(例: --wearout_rate 1.0, --wearout_rate 0.9 1.0 1.1)')
    parser.add_argument('--designed_life' , type=int  , default=1000000, help='部品ライフ設計値。算術平均やB(10)ライフなどで指定される (デフォルト: 1000000)。(例: --designed_life 1000000)')
    parser.add_argument('--beta'          , type=float, default=1.0, help='βは、部品ライフをワイブル分布で表した際の形状パラメータ。β＜1で初期故障型、β=1で偶発故障型、1<βで摩耗型故障を示す (デフォルト: 1.0)。(例: --beta 1.0)')
    parser.add_argument('--eta'           , type=int  , default=None, help='ηは、部品ライフをワイブル分布で表した際の尺度パラメータ。 (デフォルト: 部品ライフ設計値)。(例: --eta 100000)')
    parser.add_argument('--check_interval', type=str  , default='60*24*10', help='保守計画における保守間隔 (単位 [分]) (デフォルト: 60*24*10 (10日間の意味))。(例: --check_interval 60*24*10)')
    parser.add_argument('--maxt'          , type=str  , default='60*24*30*12', help='シミュレーション期間 (単位 [分]) (デフォルト: 60*24*30*12 (1年間の意味))。(例: --maxt 60*24*30*12)')
    parser.add_argument('--maxx'          , type=int  , default=200, help='交換部品数の最大値。この指定に達した時点でシミュレーションを終了する (デフォルト: 200)。(例: --maxx 200)')
    parser.add_argument('--iter'          , type=int  , default=1, help='シミュレーション回数 (デフォルト: 1)。(例: --iter 10)')
    parser.add_argument('--seed'          , type=int, default=None, help='random.seed() 初期値。(デフォルト: None)。(例: --seed 42)')

    args = parser.parse_args()
    args.maxt = eval(args.maxt)
    args.check_interval = eval(args.check_interval)

    if args.eta is None:
        args.eta = args.designed_life

    wearout_rate_list = []
    for x in args.wearout_rate:
        float_x = float(x)
        assert 0.0 < float_x <= 3.0, f'管理目標 --wearout_rate は 0.0 を超える、3.0 以下の float 値を 1つ以上指定する。異常値: {float_x}'
        wearout_rate_list.append(float_x)

    assert 1   <= args.designed_life      , f'設計値 --designed_life は 1 以上の int 値を指定する'
    assert 0.0 <  args.beta               , f'部品ライフの形状パラメータ --beta は 0.0 < beta の float 値を指定する'
    assert 1   <= args.eta                , f'部品ライフの尺度パラメータ --eta は 1 <= eta 以上の int 値を指定する'
    assert 1   <= args.check_interval     , f'保守間隔 --check_interval  は 1 以上の値となる数値、あるいは計算式を指定する。(例: --check_interval 60*24*10)'
    assert 1   <= args.maxt               , f'シミュレーション期間 --maxt は 1 以上の値となる数値、あるいは計算式を指定する。(例: --maxt 60*24*30*12)'
    assert 1   <= args.maxx               , f'交換部品数の最大値 --maxx は 1 以上の値となる数値を指定する。(例: --maxx 200)'
    assert 1   <= args.iter               , f'シミュレーション回数 --iter は 1 以上の数値を指定する。(例: --iter 10)'
    assert (args.seed is None) or isinstance(args.seed, int), f'random.seed() 初期値 --seed は int 値を指定する。(例: --seed 42)'

    # print(f'args={args}')
    # sys.exit(1)

    # args は固定したい。シミュレーションにパラメータを引き継ぐため dict 様の params を作成する
    params = Dict()  # Dict() パッケージはドット記法が可能
    params.step           = args.step
    params.debug          = args.debug
    params.wearout_rate   = args.wearout_rate
    params.designed_life  = args.designed_life
    params.beta           = args.beta
    params.eta            = args.eta
    params.check_interval = args.check_interval
    params.maxt           = args.maxt
    params.maxx           = args.maxx
    params.iter           = args.iter
    params.seed           = args.seed

    # print(f'params={params}')
    # sys.exit(1)

    return args
# end-of def arg_parse

def print_t(env, s):
    # print(f't={env.now:3d}: {s}')
    # print(f't={env.now:.2f}: {s}')
    if params.debug:
        print(f't={env.now:.2f}: {int(env.now/(60*24))}日 {s}')

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

class PrintJob():
    '''印刷ジョブ'''
    MAX_PAGE_LENGTH = 2000  # 印刷ジョブページ長の最大 (最小は1)
    MAX_SET_PER_JOB = 2000  # 印刷部数の最大 (最小は1)

    def generate_customer_print_job(self):
        '''顧客の未知パラメータに基づく印刷ジョブを作成'''
        # トータルエリアカバレッジ
        area_coverage_list = [0.10, 0.10]  # [平均(mu), 分散(sigma)]
        mu, sigma = area_coverage_list
        area_coverate = my_gauss(mu, sigma, upper_limit=2000, number_of_digits=2)

        # 印刷用紙サイズ
        customer_printed_matters = {'葉書': 0.0, 'A4': 1.0, 'B4':0.0, 'A3':0.0, '長尺':0.0} # サイズ:割合
        paper_size = random.choices(
                list(customer_printed_matters.keys()),
                weights = list(customer_printed_matters.values())
        )[0]

        # 印刷ジョブ長さ
        page_length_list = [300, 300]  # [平均(mu), 分散(sigma)]
        page_length_mu, page_length_sigma = page_length_list
        page_length = int(my_gauss(page_length_mu, page_length_sigma, self.MAX_PAGE_LENGTH, 0))

        # 両面/片面
        duplex_rate_list = [0.5, 0.3]  # [平均(mu), 分散(sigma)]
        duplex_rate_mu, duplex_rate_sigma = duplex_rate_list
        self.duplex_rate = my_gauss(duplex_rate_mu, duplex_rate_sigma, 1.0, 2)
        if 0.5 <= self.duplex_rate:
            duplex_or_simplex = 'duplex'
        else:
            duplex_or_simplex = 'simplex'

        return (area_coverate, paper_size, page_length, duplex_or_simplex)

    def __init__(self, env, id, area_coverage=0.1, paper_size='A4', page_length=1, duplex_or_simplex='simplex'):
        '''印刷ジョブ作成'''
        self.id            = id
        self.env           = env
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
      部品ライフを [ページ] で表現する。用紙サイズの異なりを補正するため、[ページ] は「A4長辺」を1とする。
    '''

    # 用紙長比
    paper_length_ratio = {
        '葉書' : 148/210,    # 葉書長辺 / A4短辺 (タテ置き)
        'A4'   : 210/210,    #   A4短辺 / A4短辺 (ヨコ置き)
        'B4'   : 364/210,    #   B4長辺 / A4短辺 (タテ置き)
        'A3'   : 483/210,    #   A3長辺 / A4短辺 (タテ置き)
        '長尺' : 1200/210,   # 長尺長辺 / A4短辺 (タテ置き)
    }

    parts_life = None  # 偽のライフ実績 (list)
 
    def set_part_life_distribution(self):
        '''部品ライフ分布を生成(ワイブル分布)'''
        # (1) 正規分布
        # return int(random.gauss(1000000, 100000))  # 正規分布は負の部品ライフを生成するため適当でない
        # 
        # (2) ワイブル分布 (部品ライフ実績から推定する場合を想定。このライブラリは打ち切りを含むデータに対応している。)
        # if self.parts_life is None:
        #     def get_dummy_failures():
        #         '''(仮の) 部品ライフ実績を生成する (ここではワイブル分布からサンプリングした(20件))'''
        #         wd = Weibull_Distribution(alpha=params.eta, beta=params.beta)
        #         # wd.plot()  # 全関数 (PDF, CDF, SF, HF, CHF) の表示
        #         parts_life = wd.random_samples(20)
        #         if self.env:
        #             print_t(self.env, f'eta={params.eta} beta={params.beta} parts_life={parts_life}')
        #         return parts_life
        #     parts_life = get_dummy_failures()  # (仮の) 部品ライフ実績を生成する
        # 
        #     # ライフ実績をもとにワイブル分布を生成 (打ち切りはなし(right_censored=None)と仮定した)
        #     self.wd = Fit_Weibull_2P(failures=parts_life, right_censored=None, show_probability_plot=False, print_results=False).distribution
        #     self.wd.plot()  # 全関数 (PDF, CDF, SF, HF, CHF) の表示
        # 
        # (3) ワイブル分布 (alpha, beta が既知の場合)
        if self.parts_life is None:
            self.wd = Weibull_Distribution(alpha=params.eta, beta=params.beta)
            if params.step:
                self.wd.plot()  # 全関数 (PDF, CDF, SF, HF, CHF) の表示  (交換時のステップ実行をしたいとき、--step 付きで実行すると交換の都度グラフが表示される)

    def __init__(self, env):
        '''交換部品の生成'''
        self.env             = env
        self.set_part_life_distribution()  # 部品ライフ分布を生成(ワイブル分布)

        self.life_limit      = int(        # 交換時の管理目標 [ページ]
            params.designed_life *         # 部品ライフ設計値 [ページ]
            params.wearout_rate            # 部品ライフ設計値を1.0とした場合の管理目標(係数)
        )
        self.cum_page_length_before = None  # 印刷ジョブ出力前の 累積印刷ページ数 [ページ]
        self.cum_page_length_after  = 0     # 印刷ジョブ出力後の 累積印刷ページ数 [ページ]

        self.survival_prob_before = None    # 印刷ジョブ出力前の生存確率

        if env:
            self.replaced_time = int(env.now)      # 交換部品を生成した日時
            print_t(self.env, f'      交換部品を設置: {self.__str__()}')

    def info(self):
        return {
            '部品ID'         : self.replaced_time,   # 部品固有のID (やっつけで交換日時を利用)
            '計画部品ライフ'   : self.life_limit,
            '累積印刷ページ数(ジョブ出力前)' : self.cum_page_length_before,
            '累積印刷ページ数(ジョブ出力後)' : self.cum_page_length_after,
        }

    def wear(self, print_job):
        '''部品ライフ進行(摩耗)
        ライフ進行の推定で利用可能な「未知パラメータ」:
          - self.area_coverage        トータルエリアカバレッジ
          - self.paper_size           用紙サイズ
          - self.page_length_before          印刷ページ長 before
          - self.page_length_after          印刷ページ長 after
          - self.duplex_or_simplex    両面片面
        '''
        # 印刷ジョブ出力前の累積印刷ページ数を保存 (def failure(self) で必要)
        self.cum_page_length_before = self.cum_page_length_after

        # 経年の計算: 「印刷ジョブページ長」×「用紙長比」を累積印刷ページ数に加算
        self.cum_page_length_after += (print_job.page_length * self.paper_length_ratio[print_job.paper_size])

        print_t(self.env, f'      累積印刷ページ数(ジョブ出力前): cum_page_length_before={self.cum_page_length_before} → 同(ジョブ出力後)cum_page_length_after={self.cum_page_length_after}')

    def failure(self):
        '''故障確率の算出'''
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

        print_t(self.env, f'      累積印刷ページ数(ジョブ出力後)={self.cum_page_length_after} (交換時の管理目標 {self.life_limit} 比: {(self.cum_page_length_after/self.life_limit*100):.2f}%) 条件付き生存確率={conditional_survival:0.5f} 一様乱数={uniform_random_number:0.5f} failure={failure} → { "故障★" if failure else "生存" }')


        # (次回の印刷ジョブ出力に備えて) 「印刷ジョブ出力後」の生存確率を保存 (処理時間を削減するため)
        self.survival_prob_before = survival_prob_after

        return failure

    def __str__(self):
      return f'[部品ID={self.replaced_time} 計画部品ライフ={self.life_limit} 累積印刷ページ数(ジョブ出力後)={self.cum_page_length_after}]'
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
        '''印刷機の予防保守のスケジュールと実施プロセス'''
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

    PRINTING_SPEED  = 30   # 印刷速度 [ページ/分]

    def __init__(self, env, id, num_printing_units=1, num_engineers=1):
        self.env = env
        self.id  = id
        self.printing_units = simpy.Resource(env, capacity=num_printing_units) # 環境にリソース追加(印刷機ユニット作成)
        self.customer_engineers = simpy.Resource(env, capacity=num_engineers)  # 環境にリソース追加(保守エンジニア)

    def terminate_simulation(self):
        # 交換部品が一定数に達したので (シミュレーション期間が残っていても) シミュレーションを終了する
        # print_t(self.env, f'len(replacement_parts_log)={len(replacement_parts_log)}\tparams.maxx={params.maxx}')
        if params.maxx == len(replacement_parts_log):
            print_t(self.env, f'交換部品数 ({len(replacement_parts_log)}) が一定数 (params.maxx={params.maxx}) に達したのでシミュレーションを終了する')
            end_event.succeed()

    def preventive_maintenance_process(self):
        '''予防保守実行プロセス'''

        print_t(self.env, f'    予防保守: BEGIN')
        # インストールされた交換部品を記録
        try:
            before_replacement_time = self.env.now                   # 交換前日時
            before_replacement_part = self.replacement_part.info()   # 交換前部品
        except AttributeError:  # 印刷機ユニット作成後、初回の部品のインストール時にこの例外が起こる (self.replacement_part が存在しないため)
            before_replacement_part = None
            pass

        # 交換部品の生成
        self.replacement_part = ReplacementPart(self.env)
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
            replacement_parts_log.append({'理由': '予防保守', '停止時間': down_time, '情報': before_replacement_part})
        # end-of if

        # 交換部品が一定数に達したので (シミュレーション期間が残っていても) シミュレーションを終了する
        self.terminate_simulation()
        print_t(self.env, f'    予防保守: END')
        # sys.exit(1)

    def corrective_maintenance_process(self):
        '''障害修理実行プロセス'''
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
        self.replacement_part = ReplacementPart(self.env)
        # 作業時間を加算
        yield self.env.timeout(random.randint(60, 90))  # raise a event  # 作業時間待機 (時間: 60-90分)

        # 停止時間(計画外ダウンタイム)の計算
        down_time = int(self.env.now - before_replacement_time)

        # 停止時間(計画外ダウンタイム)の記録
        replacement_parts_log.append({'理由': '障害修理', '停止時間': down_time, '情報': before_replacement_part})
        print_t(self.env, f'    障害修理: replacement_parts_log = {replacement_parts_log}');

        # 交換部品が一定数に達したので (シミュレーション期間が残っていても) シミュレーションを終了する
        self.terminate_simulation()
        print_t(self.env, f'    障害修理: END')
        # sys.exit(1)

    def printout_process(self, print_job):
        '''印刷実行プロセス(含む部品ライフ進行(摩耗))'''

        # 印刷ジョブの出力
        print_t(self.env, f'    印刷ジョブの出力: BEGIN {print_job}')
        yield self.env.timeout(
            print_job.page_length / self.PRINTING_SPEED
        )  # raise a event  # 印刷時間待機 (時間: 印刷ジョブ長/印刷速度)
        # 部品ライフ進行(摩耗)
        self.replacement_part.wear(print_job)
        print_t(self.env, f'    印刷ジョブの出力: END   {print_job}')

    def __str__(self):
        return f'{self.id}'
# end-of class PrintingMachine

def printing_printjob_process(env, print_job, printer):
    '''印刷ジョブ出力プロセス'''

    begin_time = env.now    # print_job の到着日時
    print_t(print_job.env, f'  印刷ジョブ到着: {print_job}')

    # 印刷機ユニットを確保
    with printer.printing_units.request() as request:
        print_t(print_job.env, f'    印刷機ユニットを確保 request開始')
        yield request  # raise a event
        print_t(print_job.env, f'    印刷機ユニットを確保 request終了')

        # 故障確率の算出と故障判断
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
        # end-of if printing_machine_failure

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
    '''印刷シミュレーションプロセス'''

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

        check_interval = params.check_interval   # 実施間隔 [日]  (例: 60*24*10 )
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

        if params.maxt <= env.now:
            print_t(env, f'シミュレーション日時 ({env.now}) が一定数 (params.maxt={params.maxt}) に達したのでシミュレーションを終了する')
            end_event.succeed()

        wait_min = 30  # [分]
        yield env.timeout(wait_min)  # raise a event  # 受注待ち待機 (時間: 30分)

        print_job_id += 1
        print_job = PrintJob(env, print_job_id)  # 印刷ジョブ生成
        env.process(printing_printjob_process(env, print_job, printer))  # 印刷ジョブ生成プロセス (平行動作)
# end-of def printingmachine_simulator_process

# 次のいずれかでシミュレーションを止める
# (1) シミュレーション日時 ({env.now}) が一定数 (params.maxt={params.maxt}) に達する
# (2) 交換部品数 ({len(replacement_parts_log)}) が一定数 (params.maxx={params.maxx}) に達する
end_event = None  # シミュレーションを終了させるイベント

# シミュレーション実行条件を表示
def simulation_parameters_str(params):
    result = (
        # f'args={args}' + '\n'\
        # f'管理目標の係数: {params.wearout_rate}' + ' ' + 
        f'部品ライフ設計値: {int(params.designed_life/1000)}k [ページ]' + ' ' + 
        f'(β={params.beta}' + ' ' + 
        f',η={int(params.eta/1000)}k)' + ' ' + 
        # f'{args.check_interval}' + ' ' + 
        # f'{args.maxt}' + ' ' + 
        # f'{args.maxx}' + ' ' + 
        # f'iter={args.iter}'
        ''
    )
    return result

# 応力-強度グラフ作成
def show_stress_strength_chart(params, wearout_rates, result_all_df):
    if len(result_all_df) == 0:
        print(f'シミュレーション終了: サンプルがなく、平均処理時間は算出できず')
        return

    # 応力-強度グラフ作成
    def plot_stress_strength_chart(each_wearout_rate):
        cond = f'管理目標(係数) = {each_wearout_rate}'
        parts_life_data = {
            'failures'      : result_all_df.loc[(result_all_df['管理目標(係数)']==each_wearout_rate) & (result_all_df['理由']=='障害修理'), '累積印刷ページ数'].tolist(), # 障害修理による交換部品ライフ
            'right_censored': result_all_df.loc[(result_all_df['管理目標(係数)']==each_wearout_rate) & (result_all_df['理由']=='予防保守'), '累積印刷ページ数'].tolist(), # 予防保守による交換部品ライフ
        }

        # print(f'cond={cond}')
        # print(f'parts_life_data={parts_life_data}')
        xvals = np.linspace(0, params.designed_life * 2, 200)

        plt.close()

        # 部品強度 (青) - parts_strength
        # --------------------------------
        plt.subplot(311)
        parts_strength_alpha = params.eta
        parts_strength_beta  = params.beta

        # '{:.0f}'.format() はエンジニアリング表記(例:1E3)を回避する
        label = f'強度 (青) [Weibull] (α=' + '{:.0f}'.format(parts_strength_alpha/1000) + 'k' + f' β={parts_strength_beta})'

        parts_strength = Weibull_Distribution(alpha=parts_strength_alpha, beta=parts_strength_beta).PDF(xvals=xvals, label=label, color='b')

        children = plt.gca().get_children()
        pm_target_line = plt.axvline(params.designed_life * each_wearout_rate, c='gray', linestyle='--')
        designed_life_line = plt.axvline(params.designed_life                , c='gray', linestyle='-')

        plt.title(f'応力-強度モデル ({cond})')
        plt.xlabel('')
        plt.ylabel('強度の確率密度')
        plt.legend( [children[0], pm_target_line, designed_life_line], [label, '予防保守の管理目標 ({:.0f}k)'.format(round(params.designed_life * each_wearout_rate/1000,0)), '設計値 ({:.0f}k)'.format(params.designed_life/1000) ]  )
        plt.xlim(0, params.designed_life * 2.1)

        # 部品交換 (赤) - simulated_failures
        # --------------------------------

        # ここでは「応力-強度モデル」に基づいて故障の確率分布を得るものとする。この確率分布として、部品ライフ分析によく使われるワイブル分布を用いる。
        # シミュレーションで得られる交換部品は、その交換理由で「障害修理」と「予防保守」に分けられる。ここでは、人為的に打ち切られた「予防保守」を除外し、部品ライフまで達した「障害修理」のみを用いた。
        # ここで「予防保守」(生存データ)を含めると元の部品ライフが再現されるだろう。

        plt.subplot(312)
        parts_exchange_dist = Fit_Weibull_2P(
            failures=parts_life_data['failures'],               # 交換データ
            # right_censored=parts_life_data['right_censored'], # 生存データ
            show_probability_plot=False,
            print_results=False
        ).distribution

        label = f'障害修理に基づく故障確率 (赤)\n[Weibull] (α=' + '{:.0f}'.format(parts_exchange_dist.alpha/1000) + 'k' + f' β={parts_exchange_dist.beta:.1f})'
        simulated_failures = parts_exchange_dist.PDF(xvals=xvals, show_plot=True, label=label, color='r')

        pm_target_line     = plt.axvline(params.designed_life * each_wearout_rate, c='gray', linestyle='--')
        designed_life_line = plt.axvline(params.designed_life                    , c='gray', linestyle='-')

        plt.title(f'')
        plt.xlabel('')
        plt.ylabel(f'故障確率')
        plt.legend()
        plt.xlim(0, params.designed_life * 2.1)

        # 応力 (緑) - parts_stress
        # --------------------------------
        plt.subplot(311).twinx()
        parts_stress = simulated_failures / parts_strength
        label = f'応力 (緑) (故障確率(赤)/強度(青)で推定)'
        plt.plot(xvals, parts_stress, label=label, color='g')
        plt.xlabel('')
        plt.ylabel('応力の確率密度')
        plt.legend(loc='center right')
        plt.xlim(0, params.designed_life * 2.1)
        max_parts_stress = max([x for x in parts_stress if not math.isnan(x)]) * 1.1  # 1.1 は上側の余白
        plt.ylim(0, max_parts_stress)

        # 部品交換ヒストグラム (ピンク/ライトブルー)
        # --------------------------------
        plt.subplot(313)
        life_df = pd.DataFrame( [('予防保守 (打切り)', x) for x in parts_life_data['right_censored']] + [('障害修理', x) for x in parts_life_data['failures']], columns=['理由', '累積印刷ページ数'])
        # print(f'life_df=\n{life_df}')
        sns.histplot(
            data=life_df,
            x='累積印刷ページ数', multiple='stack', hue='理由',
            palette=['lightblue', 'pink'],
            linewidth=.5,
        )
        plt.title('')
        plt.ylabel(f'部品数')
        plt.legend(title='シミュレーション交換理由', loc='best', labels=['障害修理', '予防保守 (打切り)'])
        plt.xlim(0, params.designed_life * 2.1)

        # グラフィック出力
        # --------------------------------
        plt.show()

    for each_wearout_rate in wearout_rates:
        plot_stress_strength_chart(each_wearout_rate)


# 交換部品数と停止時間の棒グラフ作成
def show_summary_graphics(params, wearout_rates, result_all_df):
    assert isinstance(result_all_df, pd.DataFrame) and 1 <= len(result_all_df)

    fig, axes = plt.subplots(2)
    bar_colors = {'予防保守': 'lightblue', '障害修理': 'pink'}

    bar_width = 0.7  # 0.7は棒の幅を決める定数

    try_max = result_all_df['try_i'].max()
    print(f'try_max={try_max}')

    # (1) 交換部品数を算出して棒グラフ表示
    # --------------------------------
    def create_stacked_barchart_for_exchanged_parts_number():
        nonlocal wearout_rates
        print(f'(1) 交換部品数を算出して棒グラフ表示')

        # データ作成
        pm_items = []  # (予防保守)部品数
        cm_items = []  # (障害修理)部品数
        pm_error = []  # (予防保守)エラーバー
        cm_error = []  # (障害修理)エラーバー

        for each_wearout_rate in wearout_rates:
            pm_list = []
            cm_list = []
            for i in range(0, try_max + 1):
                try_df = result_all_df.loc[ (result_all_df['管理目標(係数)'] == each_wearout_rate) & (result_all_df['try_i'] == i) ]
                # print(f'i={i} try_df=\n{try_df}')
                pm_list.append( len(try_df.loc[ try_df['理由'] == '予防保守' ]) )
                cm_list.append( len(try_df.loc[ try_df['理由'] == '障害修理' ]) )

            pm_list = [x for x in pm_list if not np.isnan(x)]
            # print(f'pm_list={pm_list}')
            cm_list = [x for x in cm_list if not np.isnan(x)]
            # print(f'cm_list={cm_list}')

            pm_items.append(round(np.mean(pm_list),1))
            cm_items.append(round(np.mean(cm_list),1))

            pm_error.append( np.std(pm_list, ddof=1)/np.sqrt(len(pm_list)) )
            cm_error.append( np.std(cm_list, ddof=1)/np.sqrt(len(cm_list)) )

        exchange_parts = {
            '予防保守': [pm_items, pm_error],
            '障害修理': [cm_items, cm_error],
        }
        print(f'exchange_parts = {exchange_parts}')

        if len(wearout_rates) == 1:
            width = 1.0
        else:
            width = round( (max(wearout_rates)-min(wearout_rates))/len(wearout_rates) * bar_width, 2)
        bottom = np.zeros(len(wearout_rates))
        for reason, [exchange_part, y_err] in exchange_parts.items():
            print(f'(reason, exchange_part, y_err=)={(reason, exchange_part, y_err)}')
            p = axes[0].bar(wearout_rates, exchange_part, width, label=reason, bottom=bottom, color=bar_colors[reason], yerr=y_err, ecolor='gray')
            bottom += exchange_part
            axes[0].bar_label(p, label_type='center')
        axes[0].set_title(f'交換部品数\n{simulation_parameters_str(params)}' + (" (エラーバーは標準誤差)" if 1 < params.iter else ""))
        # axes[0].set_xlabel('管理目標(係数)')
        axes[0].set_ylabel('交換部品数')
        axes[0].legend(title="理由", loc='upper right', reverse=True)
    create_stacked_barchart_for_exchanged_parts_number()

    # (2) 停止時間を算出して棒グラフ表示
    # --------------------------------
    def create_stacked_barchart_for_wearout_rates():
        nonlocal wearout_rates
        print('(2) 停止時間を算出して棒グラフ表示')

        # データ作成
        pm_items = []  # (予防保守)停止時間
        cm_items = []  # (障害修理)停止時間
        pm_error = []  # (予防保守)エラーバー
        cm_error = []  # (障害修理)エラーバー

        for each_wearout_rate in wearout_rates:
            pm_list = []
            cm_list = []
            for i in range(0, try_max + 1):
                try_df = result_all_df.loc[ (result_all_df['管理目標(係数)'] == each_wearout_rate) & (result_all_df['try_i'] == i) ]
                # print(f'i={i} try_df=\n{try_df}')
                pm_list.append( try_df.loc[ try_df['理由'] == '予防保守', '停止時間' ].sum() )
                cm_list.append( try_df.loc[ try_df['理由'] == '障害修理', '停止時間' ].sum() )

            pm_list = [x for x in pm_list if not np.isnan(x)]
            # print(f'pm_list={pm_list}')
            cm_list = [x for x in cm_list if not np.isnan(x)]
            # print(f'cm_list={cm_list}')

            pm_items.append(round(np.mean(pm_list),1))
            cm_items.append(round(np.mean(cm_list),1))

            pm_error.append( np.std(pm_list, ddof=1)/np.sqrt(len(pm_list)) )
            cm_error.append( np.std(cm_list, ddof=1)/np.sqrt(len(cm_list)) )

        stop_times = {
            '予防保守': [pm_items, pm_error],
            '障害修理': [cm_items, cm_error],
        }
        print(f'stop_times = {stop_times}')

        if len(wearout_rates) == 1:
            width = 1.0
        else:
            width = round( (max(wearout_rates)-min(wearout_rates))/len(wearout_rates) * bar_width, 2)
        bottom = np.zeros(len(wearout_rates))
        for reason, [stop_time, y_err] in stop_times.items():
            # print(f'(reason, stop_time, y_err=)={(reason, stop_time, y_err)}')
            p = axes[1].bar(wearout_rates, stop_time, width, label=reason, bottom=bottom, color=bar_colors[reason], yerr=y_err, ecolor='gray')
            bottom += stop_time
            axes[1].bar_label(p, label_type='center')
        axes[1].set_title('停止時間')
        axes[1].set_xlabel('管理目標(係数)')
        axes[1].set_ylabel('停止時間')
        axes[1].legend(title="理由", loc='upper right', reverse=True)
    create_stacked_barchart_for_wearout_rates()

    # (最後) グラフ出力
    # ================================
    plt.show()

def main():
    global args
    args = arg_parse()

    # シミュレーション実行
    def do_simurations():
        # シミュレーション実行
        global end_event

        global wait_times, printing_jobs_log, replacement_parts_log
        wait_times = []             # print_job 毎の印刷所要時間
        printing_jobs_log = []      # print_job 毎の終了時刻と成否
        replacement_parts_log = []  # 交換した部品 [交換理由, 停止時間, 部品情報]

        random.seed(args.seed)      # 乱数発生器の初期化 (デフォルト値 None はシステム時刻使用)

        num_printing_units, num_engineers = [1, 1]

        env = simpy.Environment()  # 環境作成
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
        # print(f'    停止時間              : {downtime_dict}')  # downtime_dict = {'予防保守': {'停止時間': 30, '交換部品数': 1}, '障害修理': {'停止時間': 85, '交換部品数': 1}}
        return downtime_dict

    # 管理目標リスト (部品ライフ設計値にかかる係数)
    wearout_rates = args.wearout_rate

    global result_all_df

    # 印刷シミュレーションを実行して、部品交換リストを返す
    def simulate_each_management_target(wearout_rate):
        '''印刷シミュレーションを実行して、部品交換リスト result_all_df を返す'''
        result_all = []

        for wearout_rate in wearout_rates:
            print(f'wearout_rate={wearout_rate}')

            params.wearout_rate = wearout_rate

            # 同じ条件でシミュレーションを繰り返し、その結果を result_all に追記する
            for try_i in range(params.iter):   # 何回繰り返すか?
                print(f'  wearout_rate={wearout_rate} try_i={try_i}')

                print(f'    シミュレーション開始')
                results_dict = do_simurations()               # シミューレションを行う
                downtime_dict = summarize_simulation_results(results_dict['replacement_parts_log'])   # シミュレーション結果を要約
                print(f'    シミュレーション終了: {downtime_dict}')

                replacement_parts_log = results_dict['replacement_parts_log']
                for item in replacement_parts_log:
                    result_all.append([
                        wearout_rate, # 管理目標(係数)
                        try_i,        # 繰り返し番号
                        item['理由'],
                        item['停止時間'],
                        item['情報']['累積印刷ページ数(ジョブ出力後)']
                    ])
                # end-of for item in replacement_parts_log
            # end-of for i in range
        # end-of wearout_rate in 

        result_all_df = pd.DataFrame(result_all, columns=['管理目標(係数)', 'try_i', '理由', '停止時間', '累積印刷ページ数'])
        return result_all_df

    result_all_df = simulate_each_management_target(wearout_rates)
    # print(f'result_all_df=\n{result_all_df}')

    # 応力-強度グラフ作成
    show_stress_strength_chart(params, wearout_rates, result_all_df)
 
    # 交換部品数と停止時間の棒グラフ作成
    show_summary_graphics(params, wearout_rates, result_all_df)

    def show_parameters():
        '''シミュレーション実行条件を表示'''
        result = \
            f'args={args}' + '\n'\
            f'予防保守の管理目標(係数)  args.wearout_rate   = {args.wearout_rate}' + '\n'\
            f'部品ライフ設計値          args.designed_life  = {args.designed_life} [ページ]' + '\n'\
            f'部品ライフ形状パラメータ  args.beta           = {args.beta}' + '\n'\
            f'部品ライフ尺度パラメータ  args.eta            = {args.eta}' + '\n'\
            f'保守間隔                  args.check_interval = {args.check_interval}' + '\n'\
            f'シミュレーション期間      args.maxt           = {args.maxt} [分] (= {args.maxt/(60*24)} [日])' + '\n'\
            f'交換部品数の最大値        args.maxx           = {args.maxx}' + '\n'\
            f'シミュレーション回数      args.iter           = {args.iter}' + '\n'\
            f'random.seed() 初期値      args.seed           = {args.seed} ({args.seed if args.seed else "システム時刻使用"})'
        return result
    print(show_parameters())
# end-of def main

if __name__ == '__main__':
    main()
