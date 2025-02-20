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
import seaborn.objects as so
from addict import Dict

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P

wait_times = None             # print_job 毎の印刷所要時間
printing_jobs_log = None      # print_job 毎の終了時刻と成否
replacement_parts_log = None  # 交換した部品 [交換理由, 停止時間, 部品情報]

def arg_parse():
    global params

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'         , action='store_true', default=False)
    parser.add_argument('--wearout_rate'  , type=float, default=1.0, help='予防保守における管理目標。部品ライフ設計値を1.0とした場合の消耗率を指定する。 (デフォルト: 1.0)。(例: --wearout_rate 1.0)')
    parser.add_argument('--designed_life' , type=int  , default=1000000, help='部品ライフ設計値。算術平均やB(10)ライフなどで指定される (デフォルト: 1000000)。(例: --designed_life 1000000)')
    parser.add_argument('--beta'          , type=float, default=1.0, help='βは、部品ライフをワイブル分布で表した際の形状パラメータ。β＜1で初期故障型、β=1で偶発故障型、1<βで摩耗型故障を示す (デフォルト: 1.0)。(例: --beta 1.0)')
    parser.add_argument('--eta'           , type=int  , default=None, help='ηは、部品ライフをワイブル分布で表した際の尺度パラメータ。 (デフォルト: 部品ライフ設計値)。(例: --eta 100000)')
    parser.add_argument('--check_interval', type=str  , default='60*24*10', help='保守計画における保守間隔 (単位 [分]) (デフォルト: 60*24*10 (10日間の意味))。(例: --check_interval 60*24*10)')
    parser.add_argument('--maxt'          , type=str  , default='60*24*30*12', help='シミュレーション期間 (単位 [分]) (デフォルト: 60*24*30*12 (1年間の意味))。(例: --maxt 60*24*30*12)')
    parser.add_argument('--maxx'          , type=int  , default=200, help='交換部品数の最大値。この指定に達した時点でシミュレーションを終了する (デフォルト: 200)。(例: --maxx 200)')
    parser.add_argument('--iter'          , type=int  , default=1, help='シミュレーション回数 (デフォルト: 1)。(例: --iter 10)')

    args = parser.parse_args()
    args.maxt = eval(args.maxt)
    args.check_interval = eval(args.check_interval)

    if args.eta is None:
        args.eta = args.designed_life

    assert 0.0 < args.wearout_rate <= 3.0 , f'管理目標 --wearout_rate は 0.0 < wearout_rate <= 3.0 の float 値を指定する'
    assert 1   <= args.designed_life      , f'設計値 --designed_life は 1 以上の int 値を指定する'
    assert 0.0 <  args.beta               , f'形状パラメータ --beta は 0.0 < beta の float 値を指定する'
    assert 1   <= args.eta                , f'度パラメータ --eta は 1 <= eta 以上の int 値を指定する'
    assert 1   <= args.check_interval     , f'保守間隔 --check_interval  は 1 以上の値となる数値、あるいは計算式を指定する。(例: --check_interval 60*24*10)'
    assert 1   <= args.maxt               , f'シミュレーション期間 --maxt は 1 以上の値となる数値、あるいは計算式を指定する。(例: --maxt 60*24*30*12)'
    assert 1   <= args.maxx               , f'交換部品数の最大値 --maxx は 1 以上の値となる数値を指定する。(例: --maxx 200)'
    assert 1   <= args.iter               , f'シミュレーション回数 --iter は 1 以上の数値を指定する。(例: --iter 10)'

    # print(f'args={args}')
    # sys.exit(1)

    # args は固定したい。シミュレーションにパラメータを引き継ぐため dict 様の params を作成する
    params = Dict()  # Dict() パッケージはドット記法が可能
    params.debug          = args.debug
    params.wearout_rate   = args.wearout_rate
    params.designed_life  = args.designed_life
    params.beta           = args.beta
    params.eta            = args.eta
    params.check_interval = args.check_interval
    params.maxt           = args.maxt
    params.maxx           = args.maxx
    params.iter           = args.iter

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

    # DESIGNED_LIFE = params.designed_life                   # [ページ] 部品ライフ設計値
    # LIFE_LIMIT = int(DESIGNED_LIFE * params.wearout_rate)  # [ページ] 交換時の管理目標 (部品ライフ設計値 x params.wearout_rate)

    # 用紙長比
    paper_length_ratio = {
        '葉書' : 148/210,    # 葉書長辺 / A4短辺 (タテ置き)
        'A4'   : 210/210,    #   A4短辺 / A4短辺 (ヨコ置き)
        'B4'   : 364/210,    #   B4長辺 / A4短辺 (タテ置き)
        'A3'   : 483/210,    #   A3長辺 / A4短辺 (タテ置き)
        '長尺' : 1200/210,   # 長尺長辺 / A4短辺 (タテ置き)
    }

    def get_internal_part_life(self):
        '''部品固有ライフを生成(ワイブル分布からサンプリング)'''
        # (1) 正規分布
        # return int(random.gauss(1000000, 100000))

        # (2) ワイブル分布 (alpha, betaが既知の場合)
        # wd = Weibull_Distribution(alpha=1000000, beta=2.0)
        # data = wd.random_samples(1)
        # return int(data[0])

        # (3) ワイブル分布 (ライフ実績から推定する場合)
        # wd = Weibull_Distribution(alpha=1000000, beta=2.0)
        wd = Weibull_Distribution(alpha=params.eta, beta=params.beta)
        # wd.plot()
        parts_life = wd.random_samples(20)    # ライフ実績 (ここではワイブル分布からサンプリングした(20件))
        if self.env:
            print_t(self.env, f'parts_life={parts_life}')
        fit = Fit_Weibull_2P(failures=parts_life,show_probability_plot=False,print_results=False)
        # fit.distribution.plot()
        # X_lower,X_point,X_upper = fit.distribution.CDF(CI_type='time',CI_y=0.7)
        # plt.show()
        internal_part_life = int(fit.distribution.random_samples(1)[0])   # 部品固有ライフを生成(ワイブル分布からサンプリング)
        print_t(self.env, f'      internal_part_life = {internal_part_life}')
        return internal_part_life

    def __init__(self, env):
        '''交換部品の生成'''
        self.env             = env                            # env
        self.DESIGNED_LIFE   = params.designed_life                           # [ページ] 部品ライフ設計値
        self.LIFE_LIMIT      = int(self.DESIGNED_LIFE * params.wearout_rate)  # [ページ] 交換時の管理目標 (部品ライフ設計値 x params.wearout_rate)

        self.life_limit      = self.LIFE_LIMIT                # 所定の計画部品ライフを取得 [ページ]
        self.specific_life   = self.get_internal_part_life()  # 部品固有ライフを生成(ワイブル分布からサンプリング) [ページ]
        self.cum_page_length = 0                              # 累積印刷ページ数 [ページ]

        if env:
            self.replaced_time   = int(env.now)                   # 交換部品が生成した日時
            print_t(self.env, f'      交換部品を設置: {self.__str__()}')

    def info(self):
        return {
            '部品ID'         : self.replaced_time,
            '固有部品ライフ' : self.specific_life,
            '計画部品ライフ' : self.life_limit,
            '累積印刷ページ数' : self.cum_page_length,
        }

    def wear(self, print_job):
        '''部品ライフ進行(摩耗)
        ライフ進行の推定で利用可能な「未知パラメータ」:
          - self.area_coverage        トータルエリアカバレッジ
          - self.paper_size           用紙サイズ
          - self.page_length          印刷ページ長
          - self.duplex_or_simplex    両面片面
        '''
        # 「印刷ジョブページ長」×「用紙長比」を累積印刷ページ数に加算
        self.cum_page_length += (print_job.page_length * self.paper_length_ratio[print_job.paper_size])

        print_t(self.env, f'      累積印刷ページ数: cum_page_length={self.cum_page_length}')

    def failure(self):
        '''故障確率の算出'''
        # 部品固有ライフ [ページ] <= 累積印刷ページ数 [ページ] となったら故障する
        if self.specific_life <= self.cum_page_length:
            print_t(self.env, f'故障: self.specific_life={self.specific_life} <= self.cum_page_length={self.cum_page_length}')
            # sys.exit()
            return True
        else:
            return False

    def __str__(self):
      return f'[部品ID={self.replaced_time} 固有部品ライフ={self.specific_life} 計画部品ライフ={self.life_limit} 累積印刷ページ数={self.cum_page_length}]'
# end-of class ReplacementPart

# 印刷機の保守計画
class MaintenanceWork():
    '''保守作業'''
    def __init__(self, env, printer, num_engineers=1):
        self.env = env
        self.printer = printer
        print_t(self.env, f'保守作業init')
        self.customer_engineer = simpy.Resource(env, capacity=num_engineers) # 環境にリソース追加(保守エンジニア)
    # end-of def __init__

    def preventive_maintenance_setup_process(self, check_interval):
        '''印刷機の予防保守のスケジュールと実施プロセス'''
        def local_print_t(s):
            print_t(self.env, s)
            pass
        local_print_t(f'■(予防保守)計画: BEGIN : {self.printer.replacement_part}')

        next_preventive_maintenance_time = self.env.now + check_interval  # 次回の予防保守の予定日
        local_print_t(f'■(予防保守)待機: 次回check t = {next_preventive_maintenance_time}')

        # 次回の予防保守の日時が来るまで待機
        yield self.env.timeout(check_interval)   # 次回の予防保守まで待機 (時間: check_interval)

        # 現在部品ライフが計画部品ライフを超えているかいないか判断
        page_length_diff = (
            self.printer.replacement_part.cum_page_length - 
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

                local_print_t(f'■(予防保守)      エンジニア開放')
            # end-of with
            local_print_t(f'■(予防保守)交換: {self.printer.replacement_part}')
        # end-of if 

        # 次回の予防保守 (今回、交換しても交換しなくても、次回の計画を要する)
        self.env.process(self.preventive_maintenance_setup_process(check_interval))  # 印刷機の予防保守のスケジュールと実施プロセス
        
        local_print_t('■(予防保守)完了: END')
    # end-of def preventive_maintenance_setup_process
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

    print_t(print_job.env, f'  印刷ジョブ終了: {print_job}    succeeds = {succeeds}')
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

        # check_interval = 60*24*10   # 実施間隔 [日]
        check_interval = params.check_interval   # 実施間隔 [日]
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

def main():
    args = arg_parse()

    def do_simurations():
        # シミュレーション実行
        global end_event

        global wait_times, printing_jobs_log, replacement_parts_log
        wait_times = []             # print_job 毎の印刷所要時間
        printing_jobs_log = []      # print_job 毎の終了時刻と成否
        replacement_parts_log = []  # 交換した部品 [交換理由, 停止時間, 部品情報]

        random.seed(42)
        num_printing_units, num_engineers = [1, 1]

        env = simpy.Environment()  # 環境作成
        env.process( printingmachine_simulator_process(env, num_printing_units, num_engineers) )  # 印刷シミュレーションプロセス (平行動作)
        simulation_period = params.maxt    # シミュレーションを期間 params.maxt に渡って行う [単位: 分] デフォルトは1年間

        end_event = env.event() # シミュレーションを終了させるイベント
        env.run(until=end_event) # シミュレーション実行

        return (
            wait_times,             # print_job 毎の印刷所要時間
            printing_jobs_log,      # print_job 毎の終了時刻と成否
            replacement_parts_log,  # 交換した部品 [交換理由, 停止時間, 部品情報]
        )

    def count_results(results):
        ''' '''
        (   wait_times,             # print_job 毎の印刷所要時間
            printing_jobs_log,      # print_job 毎の終了時刻と成否
            replacement_parts_log,  # 交換した部品 [交換理由, 停止時間, 部品情報]
        ) = results
        # 停止時間 (計画内、計画外ダウンタイム)
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

    def show_histogram(results):
        (   wait_times,             # print_job 毎の印刷所要時間
            printing_jobs_log,      # print_job 毎の終了時刻と成否
            replacement_parts_log,  # 交換した部品 [交換理由, 停止時間, 部品情報]
        ) = results

        # 結果表示
        if len(wait_times) == 0:
            print(f'シミュレーション終了: サンプルがなく、平均処理時間は算出できず')
            return

        # 印刷所要時間
        # ------------
        # print(f'印刷所要時間(平均値)  : {statistics.mean(wait_times):.1f} [分]')

        # 印刷ジョブ
        # ------------
        # print(f'印刷ジョブ記録: {printing_jobs_log}')  # [[11.2, True], [37.3, True], ... , [518373.8, True]]

        # 交換部品
        # ------------
        # print(f'交換部品数            : {len(replacement_parts_log)} [個]')  # 交換部品数: 2
        # print(f'交換部品: {replacement_parts_log}')

        # 交換部品数ヒストグラム
        # --------------------------------
        global axes
        fig, axes = plt.subplots(2, sharex=True, sharey=True)  # X軸レンジ、Y軸レンジを共有
        axes[0].set_xlim( 0, params.designed_life * 2.0)       # 強度のサンプリングの幅が広いため、部品ライフ設計値 [ページ] の2倍で X軸レンジを打ち切った
        axes[0].set_title('交換部品数ヒストグラム')
        #axes[0].set_xlabel('累積印刷ページ数')
        axes[0].set_ylabel('部品数')

        # 強度サンプリング
        specific_lifes_list = []
        for i in range( len(replacement_parts_log) ):          # 交換部品数と同じ数のサンプル作成
            replacement_part = ReplacementPart(env=None)       # 交換部品の生成
            specific_lifes_list.append(['強度', replacement_part.specific_life])
        global partslife_df
        partslife_df = pd.DataFrame(specific_lifes_list, columns=['理由', '累積印刷ページ数'])

        # 上側にプロット
        sns.histplot(data=partslife_df, stat='count', multiple='layer', x='累積印刷ページ数', hue='理由', kde=True, element='bars', legend=True, ax=axes[0], bins=18) # bins=18 は経験的なもの
        del specific_lifes_list

        # 交換部品に基づくヒストグラム
        partslife_list = []
        for item in replacement_parts_log:
            partslife_list.append([item['理由'], item['情報']['累積印刷ページ数']])
        partslife_df = pd.DataFrame(partslife_list, columns=['理由', '累積印刷ページ数'])
        del partslife_list

        # 下側にプロット
        sns.histplot(data=partslife_df, stat='count', multiple='stack', x='累積印刷ページ数', hue='理由', kde=True,  palette='pastel', element='bars', legend=True, ax=axes[1])  # 累積('stack')

        # グラフィック描画
        plt.show()

    global result_all
    result_all = []

    # wearout_rates = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    # wearout_rates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    wearout_rates = [i/100 for i in range(40, 161, 5)]   # [0.4, 0.45, 0.5, ...  1.5,  1.55,  1.6]  (25 items)
    for wearout_rate in wearout_rates:
        print(f'wearout_rate={wearout_rate}')

        params.wearout_rate = wearout_rate

        # 同じ条件でシミュレーションを繰り返し、each_downtime に記録する
        global each_downtime
        each_downtime = {
            '予防保守': {'停止時間': [], '交換部品数': []},
            '障害修理': {'停止時間': [], '交換部品数': []},
        }
        for i in range(params.iter):   # 何回繰り返すか?
            print(f'  wearout_rate={wearout_rate} i={i}')

            print(f'    シミュレーション開始')
            results = do_simurations()

            downtime_dict = count_results(results)

            print(f'    シミュレーション終了: {downtime_dict}')

            # show_histogram(results)  # 遅い (多数のグラフを表示するため)
            replacement_parts_log = results[2]

            for item in replacement_parts_log:
                result_all.append([
                    wearout_rate, # '消耗率'
                    item['理由'],
                    item['停止時間'],
                    item['情報']['累積印刷ページ数']
                ])
            # end-of for item
        # end-of for i in range
    global result_all_df
    result_all_df = pd.DataFrame(result_all, columns=['消耗率', '理由', '停止時間', '累積印刷ページ数'])

    # ================================
    # グラフィック作成
    # ================================
    fig, axes = plt.subplots(2)
    bar_colors = {'予防保守': 'lightblue', '障害修理': 'pink'}

    # --------------------------------
    # (1) 交換部品数を算出
    # --------------------------------
    exchange1_list = []  # 予防保守
    exchange2_list = []  # 障害修理

    for each_wearout_rate in wearout_rates:
        each_result_all_df = result_all_df.loc[ result_all_df['消耗率'] == each_wearout_rate ]
        print(f'each_wearout_rate={each_wearout_rate}	each_result_all_df=\n{each_result_all_df}')

        preventive_maintenance_count = len(each_result_all_df.loc[ each_result_all_df['理由'] == '予防保守' ])
        corrective_maintenance_count = len(each_result_all_df.loc[ each_result_all_df['理由'] == '障害修理' ])
        exchange1_list.append(preventive_maintenance_count)
        exchange2_list.append(corrective_maintenance_count)

    wearout_rates = result_all_df['消耗率'].unique().tolist()
    print(f'wearout_rates = {wearout_rates}')
    exchange_parts = {
        '予防保守': exchange1_list,
        '障害修理': exchange2_list,
    }
    print(f'exchange_parts = {exchange_parts}')

    width = round( (max(wearout_rates)-min(wearout_rates))/len(wearout_rates) * 0.8, 2)
    bottom = np.zeros(len(wearout_rates))
    for reason, exchange_part in exchange_parts.items():
        print(f'(reason, exchange_part)={(reason, exchange_part)}')
        p = axes[0].bar(wearout_rates, exchange_part, width, label=reason, bottom=bottom, color=bar_colors[reason])
        bottom += exchange_part
        axes[0].bar_label(p, label_type='center')
    # axes[0].set_title('交換部品数')
    axes[0].set_xlabel('管理指標 - 消耗率')
    axes[0].set_ylabel('交換部品数')
    axes[0].legend(title="理由", loc='upper right', reverse=True)

    # --------------------------------
    # (2) 停止時間を算出
    # --------------------------------
    downtime1_list = []  # 予防保守
    downtime2_list = []  # 障害修理

    for each_wearout_rate in wearout_rates:
        each_result_all_df = result_all_df.loc[ result_all_df['消耗率'] == each_wearout_rate ]
        print(f'each_wearout_rate={each_wearout_rate}	each_result_all_df=\n{each_result_all_df}')

        preventive_maintenance_downtime = each_result_all_df.loc[ each_result_all_df['理由'] == '予防保守', '停止時間'].sum()
        corrective_maintenance_downtime = each_result_all_df.loc[ each_result_all_df['理由'] == '障害修理', '停止時間'].sum()
        downtime1_list.append(preventive_maintenance_downtime)
        downtime2_list.append(corrective_maintenance_downtime)

    wearout_rates = result_all_df['消耗率'].unique().tolist()
    print(f'wearout_rates = {wearout_rates}')
    stop_times = {
        '予防保守': downtime1_list,
        '障害修理': downtime2_list,
    }
    print(f'exchange_parts = {exchange_parts}')

    width = round( (max(wearout_rates)-min(wearout_rates))/len(wearout_rates) * 0.8, 2)
    bottom = np.zeros(len(wearout_rates))
    for reason, stop_time in stop_times.items():
        print(f'(reason, stop_time)={(reason, stop_time)}')
        p = axes[1].bar(wearout_rates, stop_time, width, label=reason, bottom=bottom, color=bar_colors[reason])
        bottom += stop_time
        axes[1].bar_label(p, label_type='center')
    # axes[1].set_title('停止時間')
    axes[1].set_xlabel('管理指標 - 消耗率')
    axes[1].set_ylabel('停止時間')
    axes[1].legend(title="理由", loc='upper right', reverse=True)

    # ================================
    plt.show()

    # シミュレーション実行条件
    # ------------------------
    print(f'args={args}')
    print(f'予防保守における管理目標  args.wearout_rate   = {args.wearout_rate}')   #
    print(f'部品ライフ設計値 [ページ] args.designed_life  = {args.designed_life}')  #
    print(f'形状パラメータ            args.beta           = {args.beta}')           #
    print(f'尺度パラメータ            args.eta            = {args.eta}')            #
    print(f'保守間隔                  args.check_interval = {args.check_interval}') #
    print(f'シミュレーション期間 [分] args.maxt           = {args.maxt}')           #
    print(f'交換部品数の最大値        args.maxx           = {args.maxx}')           #
    print(f'シミュレーション回数      args.iter           = {args.iter}')           #
    sys.exit()

# end-of def main

if __name__ == '__main__':
    main()
