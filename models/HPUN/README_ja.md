# [Hybrid Pixel-Unshuffled Network for Lightweight Image Super-Resolution](https://arxiv.org/abs/2203.08921)

## 今北産業

* [EDSR](../EDSR/) に Depthwise Separatable Convolution を Residual In Residual で入れた論文
* サイズが 4 の倍数であることを仮定し、軽量になるように MaxPool でサイズを減らして計算してから Bilinear で Upsample する
* PixelUnShuffle で特徴マップの量を増やすが、MaxPool でサイズを減らすので、トータルでは計算量が 1/4 になる

## 補足

### 重みについて

* Residual In Residual の [ESRGAN](../ESRGAN/) と同じく、全体を MSRA * 0.1 の重みにしている

### 特許

* この研究は特許が取られているから、著者はコード公開してないらしい
  * [System and Method for Image Super-Resolution (US20230153946A1)](https://patents.google.com/patent/US20230153946A1/)
