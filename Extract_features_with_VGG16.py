#VGG16を使用して特徴を抽出する Extract features with VGG16

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

"""model呼び出し"""
model = VGG16(weights='imagenet', include_top=False)  # include_top: ネットワークの出力層側にある全結合層を含むかどうか 最後の直前で特徴量を抽出するのでfalse


"""画像呼び出し・modelに併せたリサイズ・array配列格納・学習時＆推論時も本来複数枚をの入力なのでその配列1次元追加"""
img_path = './images/elephant.jpg'
image_size = [224,224]

img = image.load_img(img_path, target_size=image_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


"""特徴量抽出"""
features = model.predict(x)

#print(features)
#print(features.shape)
#pritn(features.dtype)
