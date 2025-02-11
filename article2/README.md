<!-- Written in 2025 by yasuakih -->
# 【制作中】故障モデルと予防保守
この記事は、オンデマンド印刷機の定期交換部品の最適な交換時期をコンピュータシミュレーションによって決定するスタディである。

## 目的
デジタル印刷機の保守サービスを最適化するプロセスをコンピュータ上でシミュレーションを行う。この記事はプロセス全体を3つのテーマに分割した 2番目初のステップを説明する。汎用プログラミング言語のPythonとパッケージでシミュレーションを構築し、印刷機の停止時間 (ダウンタイム) と、交換される部品数をそれぞれ推定し、保守サービスによる最適な交換時期を決定する。

- <font color="gray"><a href="../article1/">顧客の未知パラメータ推定</a></font>
- 部品ライフ推定 【本記事の範囲】
- <font color="gray">機械の信頼度成長</font>

## 印刷機の保守サービスモデル

<div align="center">
  <figure>
    <img src="../img/印刷機の保守モデル.png"/>
	<br/>
    <figcaption>図. 印刷機の保守モデル</figcaption>
  </figure>
</div>

## シミュレーションの設計

### 全体の構造

<div align="center">
図2. 全体の構造
</div>

<pre><code><b>シミュレーション</b> (main)
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
    <figcaption>図. 停止時間の比較、および交換部品数の比較</figcaption>
  </figure>
</div>

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
