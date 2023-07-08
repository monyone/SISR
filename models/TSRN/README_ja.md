# [The Unreasonable Effectiveness of Texture Transfer for Single Image Super-resolution](https://arxiv.org/abs/1808.00043)

## 今北産業

* [EnhanceNet](../EnhanceNet/) のアーキテクチャを踏襲して Texture Loss (Style Loss) のみを利用する論文
* GAN を用いず MSE Loss で事前学習したあとに、全体の Texture Loss (Style Loss) を利用する
* セグメンテーションと Texture Loss (Style Loss) を組み合わせた、セグメント単位の Texture Loss を合算する損失関数も提案

## 補足

### Texture Loss (Style Loss) の取り方

* この論文では、層の最後の畳み込み + ReLU の特徴マップを利用している
  * EnhanceNet では、層の最初の畳み込み層の特徴マップを利用していた
  * なんかどっちがいいとか論文あった気がするんだけど、見つけられない...

### セグメンテーション情報の利用方法

* この論文では、データセットに MSCOCO を利用しており、このアノテーション情報から mask を作成している

### Global な Style と Local な Style

* [EnhanceNet](../EnhanceNet/) では 16x16 の小領域の Texture Loss (Style Loss) を取っていた
  * 生成結果の画像から見るに、細かい部分の書き込みが多くなる様子?
* この論文では、グローバルな Texture Loss (Style Loss) を取っている
  * 生成結果の画像から見るに、書き込みが均一になりやすい?
  * 訓練のパッチサイズに依存しそう。一応、論文中では 256x256 のパッチサイズで訓練している。
