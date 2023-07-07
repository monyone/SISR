# Accurate Image Super-Resolution Using Very Deep Convolutional Networks

## 今北産業

* Residual Learning [^ResNet] を用いて CNN の層を増やした論文
* Global Skip Connection を採用することで、Bicubic の拡大結果の差分を CNN に学習させる。
* 当時では層の数を増やすと学習が安定しなかったが、Residual Learning により 20 層での学習に成功

[^ResNet]: [CVMLエキスパートガイド: ResNet(Residual Neural Networks) とは](https://cvml-expertguide.net/terms/dl/cnn/cnn-backbone/resnet/)

## 補足

### 重みについて

* 論文内では He の初期化 (MSRA)[^init] で畳み込みの初期化をしている
  * PyTorch だと `torch.nn.init.kaiming_normal_` で初期化すればいい
  * PyTorch のデフォルトの初期化だと収束しなかったので、残差接続のネットワークでは使ったほうが良い
    * 自分の実装では fan_out で gain=2 相当の初期化をしている

[^init]: [CVMLエキスパートガイド: 重み初期化 (weight initialization): Xavier初期化とHe初期化](https://cvml-expertguide.net/terms/dl/optimization/weight-initialization/)
