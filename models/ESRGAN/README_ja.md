# [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

## 今北産業

* [SRGAN](../SRGAN/) を拡張した論文
* Redisual In Residual Dense Block を取り入れて MSE ベースでも性能が向上した
* GAN を RaGAN (Relativistic Average GAN) に変更することで学習が安定した

## 補足

### 訓練画像のサイズについて

* x4 の場合は、論文では 128x128 で訓練している
  * [SRGAN](../SRGAN/) では 96x96 だったので、ちょっと大きくなった
  * 縮小画像は 36x36 であるため、36x36 の範囲での Perceptual Loss を最適化しているといえる

### 重みについて

* 全体の畳み込みを Heの初期化に 0.1 倍に初期化すると Generator がうまく収束する
  * 論文にも MSRA の初期値を 0.1 倍するといいと書いてあった
