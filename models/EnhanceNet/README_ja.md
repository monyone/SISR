# [EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis](https://arxiv.org/abs/1612.07919)

## 今北産業

* [Image Style Transfer (CVPR 2016)](https://ieeexplore.ieee.org/document/7780634) の手法である Texture Loss (Style Loss) を SISR に適用した[^PerceptualLoss]
* Texture Loss を SISR に適用することで、抽象的なスタイルの情報を学習に利用できる
* 実際に Texture Loss を適用した結果、GAN での敵対的生成による画像の破綻を防ぎ PSNR を上げる効果が示された

[^PerceptualLoss]: Perceptual Loss の SISR への適用は [Perceptual Losses for Real-Time Style Transfer and Super-Resolution (ECCV 2016)](https://arxiv.org/abs/1603.08155) で行われており、論文内でも言及されている

## 補足

### Texture Loss (Style Loss) の算出方法

* Texture Loss は、vgg の畳み込み層の特徴マップからグラム行列を作成して計算
  * 特徴マップの [B, C, H, W] のテンソルを [B, C, H * W] につぶした上で、後ろ 2 つを転置したテンソルと掛けて [B, C, C] にする ([C, C] の行列がグラム行列)
  * 数学的理屈は [Demystifying Neural Style Transfer](https://arxiv.org/abs/1701.01036) が詳しい
* 論文内では、16x16 のパッチ毎に計算し、二乗和で合算する (MSE ではない)
  * Pytorch では unfold でパッチ化して reshape で 4 次元に整える形になる

### VGG の Mean Activation について

* 論文内では [Image Style Transfer (CVPR 2016)](https://ieeexplore.ieee.org/document/7780634) に基づいて、mean activation した vgg19 の特徴マップを用いる
  * 要するに vgg19 の 畳み込み層 の特徴マップの平均が 1 になるように 畳み込み層 の重みを変更する
  * 特徴マップの平均を 1 に正規化する際に、全体の動作に影響しないよう vgg の MaxPool を AvgPool に変更する
  * mean activation するコードは[こちら](../../data/generate_normalized_vgg19.py)
