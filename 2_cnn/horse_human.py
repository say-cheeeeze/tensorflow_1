import numpy as np
import tensorflow as tf
import os

from keras.layers.preprocessing.image_preprocessing import HORIZONTAL_AND_VERTICAL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import RMSprop

# 학습데이터 준비
train_url = 'https://storage.googleapis.com/learning-datasets/horse-or-human.zip'
file_name = 'horse-or-human.zip'
training_dir = 'horse-or-human/training'

# urllib.request.urlretrieve( train_url, file_name )

# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()

# 검증데이터 준비
validation_url = 'https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip'
validation_file_name = 'validation-horse-or-human.zip'
validation_dir = 'horse-or-human/validation'

# urllib.request.urlretrieve( validation_url, validation_file_name )

# zip_ref = zipfile.ZipFile(validation_file_name, 'r')
# zip_ref.extractall(validation_dir)
# zip_ref.close()

# 훈련과정 동안 디렉토리 순회하면서 이미지를 생성하는 반복자 객체 생성
# 디렉토리 알파벳 순서대로 레이블을 부여한다.
# 이 경우 말 이미지가 레이블0(음성 클래스), 사람 이미지가 레이블1(양성 클래스) 가 된다.
# classes 매개변수에 레이블을 부여하고 싶은 순서대로 디렉토리 이름을 나열할 수도 있다.
# train_ds 객체의 class_indices 속성에서 디렉터리 이름에 연결된 클래스 레이블을 확인할 수 있다.


# train_gen.flow_from_directory(
train_ds = tf.keras.utils.image_dataset_from_directory(

    # 대상 디렉토리
    training_dir,

    # image_size = 이미지 크기를 지정
    image_size=(300, 300),

    # 레이블 종류를 지정(이미지가 두개인 경우 binary, 두개이상일 때 categorical
    label_mode='binary'
)

# 검증 데이터 반복자 객체
validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(300, 300),
    label_mode='binary'
)

class CallBackFunc( tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.90:
            print('정확도 90% 를 달성하여 훈련을 종료합니다...')
            self.model.stop_training = True

"""
###################################################
말-사람 데이터셋을 위한 CNN 구조 convolution neural network 합성곱 신경망

1. 이미지가 300x300 이므로 많은 층이 필요하다.
2. 흑백이 아니고 컬러 이미지기 때문에 채널이 하나가 아니라 세개이다.
3. 두 종류의 이미지만 있으므로 하나의 출력 뉴런을 사용하는 이진 분류기를 만들 수 있다.
(두개의 클래스를 0과 1에 가까운 값 으로 나누어서 출력하도록 함.)
"""
model = tf.keras.models.Sequential([

    # 입력 이미지 전처리(preprocessing)
    # 케라스는 다양한 전처리 층을 제공한다.
    # 입력 이미지 크기는 300x300 이고 컬러이므로 채널이 세개 => (300,300,3)
    layers.Rescaling(1 / 255, input_shape=(300, 300, 3)),

    # 이미지 증식 추가 augmentation
    layers.RandomRotation(factor=0.2, fill_mode='nearest'),
    layers.RandomFlip(mode=HORIZONTAL_AND_VERTICAL),

    # 3x3 크기의 필터 16개를 사용한다.
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 여러개의 합성곱 층을 쌓는다. 입력 이미지가 꽤 크기 때문에
    # 특징이 강조된 작은 이미지를 많이 만들기 위해서이다.
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),

    # 마지막층은 뉴런이 한개이다.
    # 출력을 시그모이드 함수로 활성화하여 이진분류를 얻기 위함이다.
    layers.Dense(1, activation='sigmoid'),
])

# model.summary( line_length=80)
# Trainable params: 1,704,097
# 이 신경망은 파라미터 170만개를 학습하게 된다.

# 신경망을 훈련하기 위해 손실함수와 옵티마이저로 컴파일
# 클래스가 두개인 경우 사용하는 손실함수인 이진 크로스 엔트로피 binary cross entropy
# 새로운 옵티마이저인 RMSprop 에 학습속도를 제어하는 학습률 매개변수 learning_rate 지정
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

callbackFunc = CallBackFunc()

# model 훈련(훈련데이터와 레이블을 매핑하도록 훈련), epoch 마다 검증 데이터를 검증한다.
model.fit(train_ds,
          epochs=50,

          # 매 epoch 마다 모델을 테스트하기 위해 사용할 검증 데이터 지정
          validation_data=validation_ds,

          callbacks=[callbackFunc]
          )

import os

# 8개 이미지가 있다.
# hh_image_NUMBER.jpg => NUMBER : 1~4까지 말이고 5~8까지 사람인 jpg 이미지
test_img_dir = '/Users/cheeeeze/devtool/man-or-horse_img/'

test_files = os.scandir(test_img_dir)

for file in test_files:
    if file.name == '.DS_Store':
        continue

    filefullpathname = test_img_dir + file.name
    # plt.imshow(mpimg.imread(filefullpathname))
    # plt.show()

    img = tf.keras.utils.load_img(filefullpathname, target_size=(300, 300))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 이진분류를 출력하므로 클래스에 대한 점수 하나만 담겨있다.
    classes = model.predict(x)

    log_str = 'predict ======> ' + str(classes[0][0]) + ' / ' + filefullpathname

    if classes[0][0] > 0.5:
        log_str = log_str + ' ===> 사람인 것 같습니다.'
    else:
        log_str = log_str + ' ===> 말인 것 같습니다.'

    print(log_str)
