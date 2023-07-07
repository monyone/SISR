# [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)

## 今北産業

* [SRCNN](../SRCNN/) を高速化する提案
* 拡大前の画像の特徴を抽出し、PixelShuffle[^subpixel] (Sub-Pixel Convolution) で拡大しつつ再構成を行う
* PixelShuffle (Sub-Pixel Convolution) は Checkerboard Artifacts が発生しにくく[^ICNR]、後の拡大によく使われる機構になった

[^subpixel]: Transposed Convolution と表現能力は変わらない 参照: [Is the deconvolution layer the same as a convolutional layer?](https://arxiv.org/abs/1609.07009)
[^ICNR]: 未学習の場合でも発生しない初期化方法が提案されている 参照: [Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution, resize convolution and convolution resize](https://arxiv.org/abs/1707.02937)
