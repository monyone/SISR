# [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367)

## 今北産業

* [SRCNN](../SRCNN/) の開発者がモデルの高速化手法を提案
* 拡大前の画像の特徴を抽出し、Transposed Convolution [^Deconolution] で拡大しつつ再構成を行う
* ただし、Transposed Convolution は Checkerboard Artifacts が発生しやすい

[^Deconolution]: [CVMLエキスパートガイド: 転置畳み込み (TransposedConvolution, Deconolution)](https://cvml-expertguide.net/terms/dl/layers/convolution-layer/transposed-convolution/)
