import os.path
import urllib.request

from keras.src.optimizers.legacy.rmsprop import RMSProp
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


weights_url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

weights_file = 'inception_v3.h5' #87.9mb

if not ( os.path.isfile(weights_file)):
    urllib.request.urlretrieve( weights_url, weights_file)

pre_trained_model = InceptionV3( input_shape=(150,150,3),
                                 include_top=False,
                                 weights=None
                                 )

pre_trained_model.load_weights( weights_file)
pre_trained_model.summary(line_length=150)


for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('마지막 층의 출력 크기 : ', last_layer.output_shape)
# 7x7 이다.
# => 즉 이미지를 주입하면 이 층을 통과했을 때 7x7 크기의 특성 맵을 출력한다는 것임

last_output = last_layer.output
print(last_output)

# 마지막층의 출력을 밀집층에 주입하기 위해 펼친다.
x = layers.Flatten()(last_output)

# 1204개의 은닉 유닛과 렐루 활성화 함수를 사용한 완전 연결 층을 추가한다.
x = layers.Dense(1024, activation='relu')(x)

# 최종 분류 층 추가
x = layers.Dense(1,activation='sigmoid')(x)

# 이제 사전 훈련된 모델의 입력과 x 를 사용하여 모델을 정의할 수 있다.
print(x)

model = Model(pre_trained_model.input, x)

model.compile(
    optimizer=RMSProp(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['acc']
)

