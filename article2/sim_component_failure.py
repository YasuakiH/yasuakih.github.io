#!python

# sim_component_failure.py
# Written in 2025 by yasuakih

'''
About
This is a demonstration of a printing press failure model implemented using SimPy, a process-based discrete-event simulation framework in Python.

References:
強度ストレスモデル
https://reliability.readthedocs.io/en/latest/Stress-Strength%20interference.html

バスタブカーブの作成
https://reliability.readthedocs.io/en/stable/Creating%20and%20plotting%20distributions.html#example-4

usage:
python sim_component_failure.py

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

from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P

wait_times = []             # print_job 毎の印刷所要時間
printing_jobs_log = []      # print_job 毎の終了時刻と成否
replacement_parts_log = []  # 交換した部品 [交換理由, 停止時間, 部品情報]

def print_t(env, s):
    # print(f't={env.now:3d}: {s}')
    # print(f't={env.now:.2f}: {s}')
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
            self.duplex_or_simplex) = customer_print_job
        print_t(env, f'印刷ジョブを作成: {self.__str__()}')

    def __str__(self):
        return f'[#{self.id} AC{self.area_coverage} PS={self.paper_size} LEN={self.page_length} {self.duplex_or_simplex}]'
# end-of class PrintJob

class ReplacementPart():
    '''交換部品 - 交換部品の生成, 部品ライフ進行(摩耗), 故障確率の算出'''

    DESIGNED_LIFE = 1000000                   # 部品ライフ設計値
    # LIFE_LIMIT = int(DESIGNED_LIFE * 0.5)   # 設計値の50%で交換
    # LIFE_LIMIT = int(DESIGNED_LIFE * 0.6)   # 設計値の60%で交換
    # LIFE_LIMIT = int(DESIGNED_LIFE * 0.7)   # 設計値の70%で交換
    # LIFE_LIMIT = int(DESIGNED_LIFE * 0.8)   # 設計値の80%で交換
    # LIFE_LIMIT = int(DESIGNED_LIFE * 0.9)   # 設計値の90%で交換
    LIFE_LIMIT = int(DESIGNED_LIFE * 1.0)     # 設計値の100%で交換
    # LIFE_LIMIT = int(DESIGNED_LIFE * 1.5)   # 故障するまで待つ運用

    def get_internal_part_life(self):
        '''部品固有ライフを生成(ワイブル分布からサンプリング)'''
        # (1) 正規分布
        # return int(random.gauss(1000000, 100000))

        # (2) ワイブル分布 (alpha, betaが既知の場合)
        # wd = Weibull_Distribution(alpha=1000000, beta=2.0)
        # data = wd.random_samples(1)
        # return int(data[0])

        # (3) ワイブル分布 (ライフ実績から推定する場合)
        wd = Weibull_Distribution(alpha=1000000, beta=2.0)
        # wd.plot()
        parts_life = wd.random_samples(20)    # ライフ実績 (ここではワイブル分布からサンプリングした(20件))
        print_t(self.env, f'parts_life={parts_life}')
        fit = Fit_Weibull_2P(failures=parts_life,show_probability_plot=False,print_results=False)
        # fit.distribution.plot()
        # X_lower,X_point,X_upper = fit.distribution.CDF(CI_type='time',CI_y=0.7)
        # plt.show()
        return int(fit.distribution.random_samples(1)[0])   # 部品固有ライフを生成(ワイブル分布からサンプリング)

    def __init__(self, env):
        '''交換部品の生成'''
        self.env             = env
        self.replaced_time   = int(env.now)
        self.life_limit      = self.LIFE_LIMIT                # 所定の計画部品ライフを取得 [ページ]
        self.specific_life   = self.get_internal_part_life()  # 部品固有ライフを生成(ワイブル分布からサンプリング) [ページ]
        self.cum_page_length = 0                              # 累積印刷ページ [ページ]
        print_t(self.env, f'      交換部品を設置: {self.__str__()}')

    def info(self):
        return {
            '部品ID'        : self.replaced_time,
            '固有部品ライフ' : self.specific_life,
            '計画部品ライフ' : self.life_limit,
            '累積印刷ページ' : self.cum_page_length,
        }

    def wear(self, print_job):
        '''部品ライフ進行(摩耗)'''
        # 部品ライフ進行(摩耗) (累積印刷ページに「印刷ジョブページ長」を加算)
        self.cum_page_length += print_job.page_length
        print_t(self.env, f'      累積印刷ページ: cum_page_length={self.cum_page_length}')

    def failure(self):
        '''故障確率の算出'''
        # 部品固有ライフ [ページ] <= 累積印刷ページ [ページ] となったら故障する
        if self.specific_life <= self.cum_page_length:
            print_t(self.env, f'故障: self.specific_life={self.specific_life} <= self.cum_page_length={self.cum_page_length}')
            # sys.exit()
            return True
        else:
            return False

    def __str__(self):
      return f'[部品id={self.replaced_time} 固有={self.specific_life} 計画={self.life_limit} 累積={self.cum_page_length}]'
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

    def preventive_maintenance_process(self):
        '''予防保守実行プロセス'''

        print_t(self.env, '    予防保守: BEGIN')
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
            # 印刷機ユニット作成後、初回の部品のインストールの場合
            before_replacement_part = self.replacement_part.info()
        # end-of if

        # 停止時間(計画内)の記録
        replacement_parts_log.append({'理由': '予防保守', '停止時間': down_time, '情報': before_replacement_part})
        print_t(self.env, '    予防保守: END')

    def corrective_maintenance_process(self):
        '''障害修理実行プロセス'''
        print_t(self.env, '    障害修理: BEGIN')
        # インストールされた交換部品を記録
        try:
            before_replacement_time = self.env.now                   # 交換前日時
            before_replacement_part = self.replacement_part.info()   # 交換前部品
        except AttributeError:
            pass
        # 交換部品の生成
        self.replacement_part = ReplacementPart(self.env)
        # 作業時間を加算
        yield self.env.timeout(random.randint(60, 90))  # raise a event  # 作業時間待機 (時間: 60-90分)

        # 停止時間(計画外ダウンタイム)の計算
        down_time = int(self.env.now - before_replacement_time)

        # 停止時間(計画外ダウンタイム)の記録
        replacement_parts_log.append({'理由': '障害修理', '停止時間': down_time, '情報': before_replacement_part})
        print_t(self.env, '    障害修理: END')

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
    print_t(env, '印刷機ユニット作成: BEGIN')
    printing_machine_id = 'PM1'
    printer = PrintingMachine(env, printing_machine_id)
    # 印刷機ユニットを確保
    with printer.printing_units.request() as request:
        yield request
        # 部品の初回インストール
        env.process(printer.preventive_maintenance_process())  # 予防保守実行プロセス (平行動作)
    print_t(env, '印刷機ユニット作成: END')

    # 印刷機の保守計画を作成 (実施間隔: 10日)
    print_t(env, '印刷機の保守計画を作成: BEGIN')
    maintenance_work = MaintenanceWork(env, printer)
    with maintenance_work.customer_engineer.request() as request:
        yield request

        check_interval = 60*24*10   # 10 [日]

        env.process(
            maintenance_work.preventive_maintenance_setup_process(check_interval = check_interval)
        )  # 印刷機の予防保守のスケジュールと実施プロセス (平行動作)
    print_t(env, '印刷機の保守計画を作成: END')
    # sys.exit()

    # シミュレーション開始時点で存在する印刷ジョブ生成
    print_t(env, 'シミュレーション開始時点で存在する印刷ジョブ生成')
    print_job_id = 0
    initial_jobs = 1
    for print_job_id in range(initial_jobs):
        print_job = PrintJob(env, print_job_id)  # 印刷ジョブ生成
        env.process(printing_printjob_process(env, print_job, printer))  # 印刷ジョブ生成プロセス (平行動作)

    # シミュレーション期間中に受注する印刷ジョブ生成
    print_t(env, 'シミュレーション期間中に受注する印刷ジョブ生成')
    while True:   # 受注待ち
        wait_min = 30  # [分]
        yield env.timeout(wait_min)  # raise a event  # 受注待ち待機 (時間: 30分)

        print_job_id += 1
        print_job = PrintJob(env, print_job_id)  # 印刷ジョブ生成
        env.process(printing_printjob_process(env, print_job, printer))  # 印刷ジョブ生成プロセス (平行動作)
# end-of def printingmachine_simulator_process

def main():
    random.seed(42)
    num_printing_units, num_engineers = [1, 1]

    # シミュレーション実行
    print(f'シミュレーション開始')
    env = simpy.Environment()  # 環境作成
    env.process( printingmachine_simulator_process(env, num_printing_units, num_engineers) )  # 印刷シミュレーションプロセス (平行動作)
    simulation_period = 60*24*30*12  # シミュレーションを1年間行う [単位: 分]
    env.run(until=simulation_period) # シミュレーション実行

    # 結果表示
    if len(wait_times) == 0:
        print(f'シミュレーション終了: サンプルがなく、平均処理時間は算出できず')
    else:
        print(f'シミュレーション終了: 平均処理時間: {statistics.mean(wait_times):.1f} 分')
    print(f'ジョブ記録: {printing_jobs_log}')

    # ダウンタイム計算
    print(f'交換部品記録:')
    downtime_dict = {
        '予防保守': {'停止時間': 0, '交換部品数': 0},     # 計画内
        '障害修理': {'停止時間': 0, '交換部品数': 0},     # 計画外ダウンタイム
    }
    for item in replacement_parts_log:
        print(f'  {item}')
        downtime_dict[ item['理由'] ]['停止時間'] += item['停止時間']
        downtime_dict[ item['理由'] ]['交換部品数'] += 1
    print(f'downtime_dict = {downtime_dict}')

# end-of def main

if __name__ == '__main__':
    main()
