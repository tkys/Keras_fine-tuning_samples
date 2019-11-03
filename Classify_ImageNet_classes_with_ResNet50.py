# ResNet50 ImageNet クラス分類  Classify ImageNet classes with ResNet50

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


"""model呼び出し"""

model = ResNet50(weights='imagenet') # imageneの学習した重みをそのまま使う



"""画像呼び出し・modelに併せたリサイズ・array配列格納・学習時＆推論時も本来複数枚を入力もあるのでその配列1次元追加"""

img_path = './images/elephant.jpg'
image_size = [224,224]
img = image.load_img(img_path, target_size=image_size)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


"""推論　全ラベルに対しての確率　結果はリスト形式"""

preds = model.predict(x) 


"""ランキング上位Ｎ件のラベル・説明・確率"""

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)

print('Predicted:', decode_predictions(preds, top=3)[0])

## Predicted: [('n02504458', 'African_elephant', 0.7600107), ('n01871265', 'tusker', 0.19873881), ('n02504013', 'Indian_elephant', 0.04117495)]
