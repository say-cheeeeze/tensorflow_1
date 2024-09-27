import tensorflow as tf

mnist= tf.keras.datasets.fashion_mnist
(tr_img, tr_label), (test_img, test_label) = mnist.load_data()

tr_imag = tr_img / 255.0
test_img = test_img / 255.0

# tf.keras.models.Sequential 은 모델에게 훈련과 추론 기능을 제공하는 객체이다.
model = tf.keras.models.Sequential([

    # tf.keras.layers 를 이용하여 층을 정의할 수 있다.

    # 첫번째, 뉴런의 층이 아니고 입력input을 위한 크기를 지정한다.
    # 28x28 크기의 이미지이다.
    # Flatten 은 2D 배열인 행렬을 1D 배열인 벡터로 변환한다.
    # Flatten 은 입력층이라고 하지 않는다. 관례상 층으로 부르지만,
    # 입력 데이터 자체이다.
    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# model 의 방식을 지정. compile. 손실함수, 옵티마이저, 측정기준 metrics
model.compile( optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# start to train
model.fit(tr_img, tr_label)

# Returns the loss value & metrics values for the model in test mode.
evaluate = model.evaluate(test_img, test_label)
print(evaluate)

# predictions for input sample
predict = model.predict(test_img)
print(predict[0])
print(test_label[0])

