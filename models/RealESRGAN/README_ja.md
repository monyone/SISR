# [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)

## 今北産業

* [ESRGAN](../ESRGAN/) のアーキテクチャをベースとして、前処理をリアルに近づけた
* Generator は、常に4倍拡大以上するように、PixelUnShuffleをはさむようになった
* Discriminator は U-Net に近い構造になった
