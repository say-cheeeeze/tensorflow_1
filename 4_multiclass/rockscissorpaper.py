# 가위바위보 이미지 분류문제
# 다중분류 문제 multiclass classification
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

training_dir = 'rps'

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(150, 150),
    class_mode='categorical'
)

print(train_generator)

model = tf.keras.models.Sequential([
    # 합성곱층 1
    # 입력은 데이터의 크기(150,150과 일치해야하고 출력은 클래스 개수(3, 가위바위보)과 일치해야한다.
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    # 합성곱층 2
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    # 합성곱층 3
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    # 합성곱층 4
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # 밀집층에 전달하기 위해 펼친다.
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    # 다중클래스 분류 -> softmax !, 분류할 클래스의 수가 3 (가위/바위/보) => units = 3 이다.
    tf.keras.layers.Dense(3, activation='softmax')
])

# 범주형 엔트로피 손실함수를 사용해서 모델을 컴파일한다.
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])