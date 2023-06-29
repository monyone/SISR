# Image Super-Resolution Using Deep Convolutional Networks

## 今北産業

* 初めて CNN を超解像タスクに利用する SRCNN を提案
* 古典的手法 (バイキュービック) で拡大した画像を CNN を用いて修復する
* 今までの手法に対して PSNR 指標で SOTA を達成

## アーキテクチャ

* Patch extraction and representation
  * 事前拡大画像の特徴の抽出を行う
* Non-linear Mapping
  * 事前拡大画像の特徴から、修復するための高解像度向けの特徴へ変換する
  * 1x1 の畳み込みは、[^NIN] で提唱された、チャンネル方向の次元削減に利用される手法[^1x1]
* Reconstruction
  * 高解像度向けの特徴から画像を再構成する

[^NIN]: [Network In Network (2013)](https://arxiv.org/abs/1312.4400)
[^1x1]: [CVMLエキスパートガイド 1x1 畳み込み (1x1 Convolution, 点単位畳み込み層)](https://cvml-expertguide.net/terms/dl/layers/convolution-layer/1x1-convolution/)
