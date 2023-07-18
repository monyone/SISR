# [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

## 今北産業

* [EDSR](../EDSR/) に対して Residual In Residual を採用した上で Channel Attension も取り入れて性能向上を示した論文
* Channel Attension として Squeeze & Excitation でいうところの Channle Squeeze and Spatial Excitation [^SqueezeAndExcitation] を採用
* 既存の超解像タスクで SOTA を達成

[^SqueezeAndExcitation]: [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) の分類

## 補足

### 重みについて

* 公式実装だと EDSR と同じく Redisual Block の出力に定数倍 scale (0.1) を掛けて調整していた
  * でも、これだと精度が出ない感があったので Residual In Residual の [ESRGAN](../ESRGAN/) と同じく、全体を MSRA * 0.1 の重みにしている

### RCAN はアンダーフィッティングしてる?

* [Revisiting RCAN: Improved Training for Image Super-Resolution](https://arxiv.org/abs/2201.11279) によると、性能向上の余地があるらしい
  * バッチサイズを 16 -> 256 に、パッチを 48x48 -> 64x64 に、ReLU を SiLU (Swish) にすることで改善したのだとか
* かなり大きいバッチサイズなので、 Gradient Accumulation で疑似的には再現できるかな?
  * Batch Normalization がないので、基本 OK だと思うが...
* 手元で ReLU を SiLU にしても、PSNR が上がらんかったので、単純に SiLU で OK というわけでもなさそう
