# [SwiftSRGAN -- Rethinking Super-Resolution for Efficient and Real-time Inference](https://arxiv.org/abs/2111.14320)

## 今北産業

* SRResNet/SRGAN の Conv2d を SeparableConv2d に変えた論文
* SeparatableConv2d (Depthwise Separable Convolution [^DepthwiseConvolution]) でパラメータ数を削減し、速度向上と、品質が問題ないことを示した
* PyTorch 上では CuDNN 利用時は 2/3 程に計算時間が短縮された (CuDNN を OFF すると 15 倍くらいになって理論値っぽくなる)

## 補足

* Pytorch では CuDNN が有効だと Depthwise Separatable Convolution が最適化されないので、あまり効果がない様子
  * いろいろ Issue が上がっているが未解決

[^DepthwiseConvolution]: [CVMLエキスパートガイド: 深さ単位分離可能畳み込み (Depthwise Separable Convolution](https://cvml-expertguide.net/terms/dl/layers/convolution-layer/depthwise-separable-convolution/])
