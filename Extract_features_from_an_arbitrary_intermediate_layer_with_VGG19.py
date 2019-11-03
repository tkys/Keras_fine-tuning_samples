#VGG19を使用して任意の中間層から特徴を抽出する Extract features from an arbitrary intermediate layer with VGG19

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np


"""model呼び出し"""

base_model = VGG19(weights='imagenet')
#model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

# inputs=base_model.input: 入力レイヤー　vgg19の初期そのまま
# outputs=base_model.get_layer('block4_pool').output: 出力レイヤー　block4_pool を指定（最後のほう）


"""画像呼び出し・modelに併せたリサイズ・array配列格納・学習時＆推論時も本来複数枚をの入力なのでその配列1次元追加"""
img_path = './images/elephant.jpg'
image_size = [224,224]

img = image.load_img(img_path, target_size=image_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


"""特徴量抽出"""
block4_pool_features = model.predict(x)

#print(block4_pool_features)
#print(block4_pool_features.shape)
#print(block4_pool_features.dtype)
