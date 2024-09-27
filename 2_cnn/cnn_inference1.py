"""
합성곱 신경망 만들기
"""
import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

(tr_img, tr_label), (test_img, test_label) = data.load_data()

# 이미지를 정규화하기 전에 먼저, 배열에 차원을 추가해준다.
# conv2D 층은 컬러이미지를 위해 설계되었기 때문에
# 훈련 이미지도 형상을 바꿔준다. 28x28 => 28x28x1
tr_img = tr_img.reshape(60000, 28, 28, 1)
# 이미지를 정규화한다.
tr_img = tr_img / 255.0

# 테스트 데이터도 동일하게 형상 변경한다.
test_img = test_img.reshape(10000, 28, 28, 1)
test_img = test_img / 255.0

model = tf.keras.models.Sequential([

    # 64개의 합성곱 필터를 학습한다. 필터는 랜덤하게 초기화되고,
    # 시간이 지남에 따라 입력을 레이블로 매핑하기 위해 가장 좋은 필터 값을 학습한다.
    # (3,3) 은 필터의 크기이다. pooling 할 영역을 잡을 크기.
    # conv2D 층은 컬러이미지를 위해 설계되었기 때문에 세번째 차원을 1 로 지정해주어야한다.(28,28,1)
    # 컬러rgb값을 가져야하므로 세번쨰 차원은 3 이 된다.
    tf.keras.layers.Conv2D(64, (3,3),
                           activation='relu',
                           input_shape=(28,28,1)),

    # 신경망에서 풀링층은 보통 합성곱 바로 뒤에 적용한다.
    # 이미지를 나누어서 max pooling 할 사이즈 2x2 로 하겠다. 각 데이터의 최대값을 뽑아 2x2 로 줄이겠다.
    # 2x2 로 풀링의 높이와 너비가 동일한 경우 MaxPooling2D(2) 로 줄일 수 있다.
    # 아무것도 학습하지 않고 이미지 크기를 줄이기만 한다.
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D((2,2)),

    # 여기서는 Flatten층이 없고 대신 입력 크기를 지정한다.
    # 합성곱와 풀링층 다음에는 Flatten 층으로 데이터를 펼쳐서 Dense 층에 전달한다.
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model 분석해서 출력해준다.
model.summary()

# 24 epoch 에서 99% acc 를 달성한다.
# model.fit( tr_img, tr_label, epochs=50)

# model.evaluate(test_img, test_label)

# predict = model.predict(test_img)
# print(predict[0])
# print(test_label[0])


"""
/opt/anaconda3/envs/tensorflow1/bin/python /Users/cheeeeze/git/python_deeplearning1/tensorflow1/2_cnn/cnn_inference1.py 
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 64)        640       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 64)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        36928     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1600)              0         
                                                                 
 dense (Dense)               (None, 128)               204928    
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 243,786
Trainable params: 243,786
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code 0
"""