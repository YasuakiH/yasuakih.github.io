<!-- Written in 2025 by yasuakih -->
# 【制作中】定期交換部品のライフ推定による交換時期の最適化
この記事は、オンデマンド印刷機の定期交換部品の最適な交換時期をコンピュータ・シミュレーションによって推定するスタディである。

## 目的
デジタル印刷機の保守サービスを最適化するプロセスをコンピュータ上でシミュレーションを行う。この記事はプロセス全体を3つのテーマに分割した 2番目のステップを説明する。最初の記事で推定した顧客による<a href="../article1/">印刷機の使われ方</a>をもとに、定期交換部品を計画的に交換する管理目標が、印刷機の停止時間 (ダウンタイム) と交換される部品数 (コスト) に及ぼす影響を推定し、保守サービスにおける最適な交換時期を推定する。汎用プログラミング言語のPythonと無償のシミューレション用パッケージ simpy でシミュレーションを構築する。

- <font color="gray">1 顧客の未知パラメータ推定</font>
- 2 部品ライフ推定 【本記事の範囲】
- <font color="gray">3 機械の信頼度成長</font>

## 部品の故障モデル
顧客による<a href="../article1/">印刷機の使われ方</a>が部品ライフに影響を及ぼすことを表現するため、[応力-強度モデル](https://en.wikipedia.org/wiki/Stress%E2%80%93strength_analysis) (Stress–strength model) を採用した。

### 応力 - 部品の使われ方
最初の記事で推定した<a href="../article1/">印刷機の使われ方</a> は

### 強度 - 部品の強度
部品の強度にはばらつきがあるため、

(応力) の分布と、部品の強度の分布の

重なりが故障の確率とする。

部品にかかる応力によって部品が故障するという
確率的に故障する


### 応力-強度モデル

<div align="center">
  <figure>
    <a title="Cdang, CC BY-SA 3.0 &lt;https://creativecommons.org/licenses/by-sa/3.0&gt;, via Wikimedia Commons" href="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Contrainte_resistance_2d_proche.svg/551px-Contrainte_resistance_2d_proche.svg.png"><img alt="Contrainte resistance 2d proche" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Contrainte_resistance_2d_proche.svg/551px-Contrainte_resistance_2d_proche.svg.png"></a>
	<br/>
    <figcaption>図. 応力-強度モデル<br/><a href="https://commons.wikimedia.org/wiki/User:Cdang">Cdang</a>, <a href="https://creativecommons.org/licenses/by-sa/3.0">CC BY-SA 3.0</a>, via Wikimedia Commons
    </figcaption>
  </figure>
</div>


## 印刷機の保守モデル

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
  ├ <b>印刷シミュレーションプロセス</b>
  └ 結果表示
   
    <b><a href="#印刷シミュレーションプロセス">印刷シミュレーションプロセス</a></b> (printingmachine_simulator_process)
      ├ 印刷機ユニットを確保し、部品の初回インストールをするため <b>予防保守実行プロセス</b>を実行
      ├ 印刷機の保守計画を策定し (10日間隔とする)、<b>印刷機の予防保守のスケジュールと実施プロセス</b>を実行
      └ シミュレーション期間中のジョブ受注 (1年間)                                                      ← ループ
          └ 定期的(30分間隔)に<b>印刷ジョブ作成</b>し、<b>印刷ジョブの出力プロセス</b>を実行

        印刷機ユニット (class PrintingMachine)
          ├ <b><a href="#予防保守実行プロセス">予防保守実行プロセス</a></b> (preventive_maintenance_process)
          │  ├ <b>交換部品の生成</b>
          │  └ 交換作業 (待機時間: 30分)
          ├ <b><a href="#障害修理実行プロセス">障害修理実行プロセス</a></b> (corrective_maintenance_process)
          │  ├ <b>交換部品の生成</b>
          │  └ 修理作業 (待機時間: 60-90分)
          └ <b><a href="#印刷実行プロセス">印刷実行プロセス(含む部品ライフ進行(摩耗))</a></b> (printout_process)
             ├ 印刷実行 (待機時間: 印刷ジョブ長/印刷速度)
             └ <b>部品ライフ進行(摩耗)</b>

        保守作業 (class MaintenanceWork)
          └ <b><a href="#印刷機の予防保守のスケジュールと実施プロセス">印刷機の予防保守のスケジュールと実施プロセス</a></b> (preventive_maintenance_setup_process)
            ├ 次回の予防保守まで待機 (時間: 10日間)
            ├ 現在部品ライフが計画部品ライフを超過したら部品を交換
            │  ├ エンジニアおよび印刷機ユニットを確保
            │  └ <b>予防保守実行プロセス</b>
            └ 印刷機の予防保守のスケジュールと実施プロセス (次回分。再帰している)

        印刷ジョブ (class PrintJob)
          └ <b>印刷ジョブ作成</b> (init)
            └ <b><a href="#顧客の未知パラメータに基づく印刷ジョブを作成">顧客の未知パラメータに基づく印刷ジョブを作成</a></b> (generate_customer_print_job)

        <b>印刷ジョブの出力プロセス</b> (printing_printjob_process)
          ├ 印刷機ユニットを確保
          ├ <b>故障確率の算出</b>
          │  ├ 故障時、修理するエンジニアを確保
          │  └ <b>障害修理実行プロセス</b>
          ├ <b>印刷実行プロセス(含む部品ライフ進行(摩耗))</b>
          └ print_job 毎の結果を記録 (印刷所要時間, 終了時刻と成否)

            交換部品 (class ReplacementPart)
              ├ <b><a href="#交換部品の生成">交換部品の生成</a></b> (init)
              │  ├ 計画部品ライフを取得 (所定の値)
              │  └ <b>部品固有ライフを生成(ワイブル分布からサンプリング)</b> (get_internal_part_life)
              ├ <b>部品ライフ進行(摩耗)</b> (wear)
              │  └ 累積印刷ページに「ページ長」を加算し、部品ライフ進行させる
              └ <b><a href="#故障確率の算出">故障確率の算出</a></b> (failure)
                 └ 部品固有ライフ ≦ 累積印刷ページ となったら故障
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
