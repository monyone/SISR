# [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

## 今北産業

* SISR (Single Image Super Resolution) に初めて GAN を利用した論文
* GAN の Generator 側の SRResNet は、後の様々なモデルのベースになった
* 事前に SRResNet を Pixel Loss (L2) で学習しておき、 GAN の学習は Perceptual Loss + Adversarial Loss で再度学習

## 補足

### 訓練画像のサイズについて

* 論文と同じように、訓練用のパッチサイズは 96x96 か、それ以上にしないと、出力が崩れる
  * Perceptual Loss が、学習画像のサイズに依存しているため
  * 32x32 でやった場合、元画像の 8x8 px の領域から perceptual Loss を補おうとしてしまい、局所的すぎてしまった
  * 訓練用の低解像度画像が 24x24 以上 (低解像度 32x32 (128x128) でもいい結果になった) である必要あり

### Perceptual Loss について

* この論文では VGG19 の Conv5_4 を取っている
  * relu 層を通った後の post activation を利用

### Adversarial Loss について

* この論文では Adversarial Loss に 0.001 の係数を掛けている (Perceptual Loss は 1)
  * (こんな少なくていいのか疑問がある...ここらへんよくわからん...)

### 高速化について

* [Fast-SRGAN](https://github.com/HasnainRaz/Fast-SRGAN) が有名らしい
  * 以下 2 つが特徴的
    * Convolution を Depthwise Separatable Convolution へ
      * [Swift-SRGAN](../SwiftSRGAN/) とか、後続手法の高速化でもデファクト (置き換えるだけなので)
    * PixelShuffle を Bilinear Upsample + Convolution へ
      * 古典的 Upsample を使うのは [EnhanceNet](../EnhanceNet/) で見た手法だが、[ESRGAN](../ESRGAN/) でも使われてる (市民権があるらしい)
      * SwinIR では Artifact が出にくい Real-World SR と書かれていた
  * 特徴マップの数などは [Towards Real-Time Image Enhancement GANs](https://link.springer.com/chapter/10.1007/978-3-030-29888-3_15) から取っている
    * 特徴マップの数: Fast -> 32, Very-Fast -> 8
    * Residual Block の数: Fast -> 12, Very-Fast -> 16
