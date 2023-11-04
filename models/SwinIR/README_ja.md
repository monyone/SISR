# [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

## 今北産業

* Swin Transfomer を Super Resolution に使った論文


## 補足

### Swin Transfomer の使い方
* Patch Size は 1 で Embeding で B, C, H, W => B, H, W, C にするだけ
  * なので Unembeding で B, C, H, W へ戻す操作が可能
* 要するに Swin Transfomer に Channel Attension みたく入れている感じ

### 学習方法
* 帰納バイアスがないので、いまいち学習ができない印象
* 画像の用意の仕方、パッチの分割の仕方など、もうちょっと論文を読みたい

