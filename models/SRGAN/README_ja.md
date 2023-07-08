# [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

## 今北産業

* SISR (Single Image Super Resolution) に初めて GAN を利用した論文
* GAN の Generator 側の SRResNet は、後の様々なモデルのベースになった
* 事前に SRResNet を Pixel Loss (L2) で学習しておき、 GAN の学習は Perceptual Loss + Adversarial Loss で再度学習

## 補足

### Perceptual Loss について

* この論文では VGG19 の Conv5_4 を取っている
  * relu 層を通った後の post activation を利用

### Adversarial Loss について

* この論文では Adversarial Loss に 0.001 の係数を掛けている (Perceptual Loss は 1)
  * (こんな少なくていいのか疑問がある...ここらへんよくわからん...)


