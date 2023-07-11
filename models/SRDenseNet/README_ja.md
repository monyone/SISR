# [Image Super-Resolution Using Dense Skip Connections](https://ieeexplore.ieee.org/document/8237776)

## 今北産業

* DenseNet[^DenseNet] を Super Resolution に適用した論文
* すべての DenseBlock の出力を結合して BottleNeck 層に放り込む Densely Skip Connection が特徴的
* DenseBlock 内のチャンネル数が少なく、BottleNeck 層でチャンネルを削減するため、推論速度もそれなり

[^DenseNet]: [CVMLエキスパートガイド: DenseNet](https://cvml-expertguide.net/terms/dl/cnn/cnn-backbone/densenet/)
