<!-- Written in 2025 by yasuakih -->
# 【制作中】定期交換部品のライフ推定による交換時期の最適化
この記事は、オンデマンド印刷機の定期交換部品の最適な交換時期をコンピュータシミュレーションによって決定するスタディである。

## 目的
デジタル印刷機の保守サービスを最適化するプロセスをコンピュータ上でシミュレーションを行う。この記事はプロセス全体を3つのテーマに分割した 2番目のステップを説明する。最初の記事で推定した<a href="../article1/">印刷機の使われ方</a>をもとに、定期交換部品を計画的に交換する管理目標によって、印刷機の停止時間 (ダウンタイム) と交換される部品数 (コスト) に及ぼす影響を推定し、保守サービスにおける最適な交換時期を決定する。汎用プログラミング言語のPythonと無償のシミューレション用パッケージ simpy でシミュレーションを構築する。

- <font color="gray">1 顧客の未知パラメータ推定</font>
- 2 部品ライフ推定 【本記事の範囲】
- <font color="gray">3 機械の信頼度成長</font>

## 印刷機の保守モデル

### 応力-強度モデル
[応力-強度モデル](https://en.wikipedia.org/wiki/Stress%E2%80%93strength_analysis) (Stress–strength analysis)

<div align="center">
  <figure>
    <img src="img/Contrainte_resistance_2d_proche.svg"/>
	<br/>
    <figcaption>図. 応力-強度モデル
	by [Cdang](https://commons.wikimedia.org/wiki/File:Contrainte_resistance_2d_proche.svg) ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed.ja))
</figcaption>
  </figure>
</div>

### 印刷機の保守モデル

<div align="center">
  <figure>
    <img src="../img/印刷機の保守モデル.png"/>
	<br/>
    <figcaption>図. 印刷機の保守モデル (<a href="../img/印刷機の保守モデル.png" target="_blank">拡大</a>)</figcaption>
  </figure>
</div>

## シミュレーションの設計

### 全体の構造

<div align="center">
図2. 全体の構造
</div>

<pre><code>
<b>シミュレーション</b> (main)
  ├ シミュレーション環境作成
  ├ <b>印刷シミュレーションプロセス</b> (printingmachine_simulator_process)<!-- シミュレーションを1年間行う -->
  └ 結果表示
   
    <b>印刷シミュレーションプロセス</b> (printingmachine_simulator_process)
      ├ 印刷機ユニットを確保し、部品の初回インストールをする (<b>予防保守実行プロセス</b>)
      ├ 印刷機の保守計画を作成 (間隔: 10日) (印刷機の予防保守のスケジュールと実施プロセス)
      └ シミュレーション期間中、定期的(30分間隔)に<b>印刷ジョブ作成</b> (class PrintJob)(printing_printjob_process)

        印刷機ユニット
          ├ <b><a href="#予防保守実行プロセス">予防保守実行プロセス</a></b> (preventive_maintenance_process)
          │  ├ <b>交換部品の生成</b>
          │  └ 交換作業 (待機時間: 30分)
          ├ <b><a href="#障害修理実行プロセス">障害修理実行プロセス</a></b> (corrective_maintenance_process)
          │  ├ <b>交換部品の生成</b>
          │  └ 修理作業 (待機時間: 60-90分)
          └ <b>印刷実行プロセス(含む部品ライフ進行(摩耗))</b>
             ├ 印刷 (待機時間: 印刷ジョブ長/印刷速度)
             └ 部品ライフ進行(摩耗) (wear)

        保守作業
          └ <b><a href="#印刷機の予防保守のスケジュールと実施プロセス">印刷機の予防保守のスケジュールと実施プロセス</a></b> (preventive_maintenance_setup_process)
            ├ 次回の予防保守まで待機 (時間: 10日間)
            ├ 現在部品ライフが計画部品ライフを超過したら部品を交換
            │  ├ エンジニアと、印刷機ユニットを確保
            │  └ <b>予防保守実行プロセス</b>
            └ 印刷機の予防保守のスケジュールと実施プロセス (次回分) [再帰]

        <b>印刷ジョブ</b> (class PrintJob)
          └ <b><a href="#顧客の未知パラメータに基づく印刷ジョブを作成">顧客の未知パラメータに基づく印刷ジョブを作成</a></b> (generate_customer_print_job)

        印刷ジョブの出力プロセス (printing_printjob_process)
          ├ 印刷機ユニットを確保
          ├ 故障確率の算出と故障判断
          │  ├ 故障時、修理するエンジニアを確保
          │  └ <b>障害修理実行プロセス</b>
          ├ <b><a href="#印刷実行プロセス">印刷実行プロセス(含む部品ライフ進行(摩耗))</a></b> (printout_process)
          └ print_job 毎の結果を記録 (印刷所要時間, 終了時刻と成否)

            <b><a href="#交換部品の生成">交換部品の生成</a></b>
              ├ 交換部品の生成 (init)
              │  ├ 所定の計画部品ライフを取得
              │  └ <b>部品固有ライフを生成(ワイブル分布からサンプリング)</b> (get_internal_part_life)
              ├ <b>部品ライフ進行(摩耗)</b> (累積印刷ページに「ページ長」を加算) (wear)
              └ <b>故障確率の算出</b>部品固有ライフ [ページ] ≦ 累積印刷ページ [ページ] となったら故障する (failure)
</code></pre>

## 実験結果
次のコマンドラインを用いてシミュレーションを実施した。

``` shell
python TBD
```

### 停止時間

### 交換部品数

<div align="center">
  <figure>
    <img src="img/sim_component_failure-成績.png"/>
	<br/>
    <figcaption>図. 定期交換部品の計画的な交換時期が、印刷機の停止時間と交換部品数に及ぼす影響</figcaption>
  </figure>
</div>

## 課題
### 保守作業員コストの反映

### リアリティ向上
複数部品の同時交換

## 結論

## 付録
### ソースコード
* [sim_component_failure.py](sim_component_failure.py)

### コマンドライン
``` shell
TBD
```

----
このページに掲載した作品 (テキスト、プログラムコードなど) はパブリック・ドメインに提供しています。詳細は [CC0 1.0 全世界 コモンズ証](https://creativecommons.org/publicdomain/zero/1.0/deed.ja) をご覧ください。
