# Keras_fine-tuning_samples

Kerasの事前学習した重みを利用した予測・特徴量抽出・fine-tuningのサンプル


from
https://keras.io/ja/applications/#xception





モデルをインスタンス化すると重みは自動的にダウンロードされます．重みは~/.keras/models/に格納されます．

利用可能なモデル
ImageNetで学習した重みをもつ画像分類のモデル:

```
Xception
VGG16
VGG19
ResNet50
InceptionV3
InceptionResNetV2
MobileNet
DenseNet
NASNet
MobileNetV2
```

これら全てのアーキテクチャは全てのバックエンド（TensorFlowやTheano，CNTK）と互換性があり，モデルはインスタンス化する時はKerasの設定ファイル~/.keras/keras.jsonに従って画像のデータフォーマットが設定されます． 

例えば，image_dim_ordering=channels_lastとした際は，このリポジトリからロードされるモデルは，TensorFlowの次元の順序"Height-Width-Depth"にしたがって構築されます．
