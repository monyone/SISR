# [The Unreasonable Effectiveness of Texture Transfer for Single Image Super-resolution](https://arxiv.org/abs/1808.00043)

## 今北産業

* [EnhanceNet](../EnhanceNet/) のアーキテクチャを踏襲して Texture Loss (Style Loss) を利用した論文
* GAN を利用せずとも、GAN より良い精度を出せる???
* (いまいち読み込めてなくて、EnhanceNet よりよくなる点がわからない)

## 補足

### Texture Loss (Style Loss) の取り方

* この論文では、層の最後の畳み込み + ReLU の特徴マップを利用している
  * EnhanceNet では、層の最初の畳み込み層の特徴マップを利用していた
  * なんかどっちがいいとか論文あった気がするんだけど、見つけられない...
