import tensorflow as tf

fashion_data = tf.keras.datasets.fashion_mnist
(tr_img, tr_label), (test_img, test_label) = fashion_data.load_data()

# 정규화
tr_img = tr_img / 255.0
test_img = test_img / 255.0

"""
1. Flatten : 뉴런의 층이 아니라 입력을 위한 크기를 지정한다. flatten 층은 2D 배열인 행렬을 1D 배열인 벡터로 변환한다.
2. Dense : 뉴런의 층. hidden layer, 128개의 뉴런을 지정(임의로 개수 지정), 랜덤하게 초기화됨 
3. Dense : 찾아야할 클래스가 10개이므로 10개의 뉴런을 두는 Dense 층.
           softmax 활성화 함수를 사용. 
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


def training( epoch ):
    """
    epoch 5 fit 을 진행하고 predict 데이터 확인해보기
    """

    # 훈련 세트와 테스트 세트를 가져온다. 6만개의 훈련 이미지와 1만개의 테스트 이미지 세트
    # 총 7만개의 이미지+레이블 세트

    # 정규화한다. 정규화는 성능을 높인다.
    # 이미지 픽셀이 모두 흑백이므로 0~255 값을 가지는데, 255 로 나누면 각 픽셀값을 0 ~ 1 사이의 값으로 나타낼 수 있다.

    # 모델 컴파일. 방식을 지정한다.
    # 손실함수와 옵티마이저
    # 희소한 범주형 크로스 엔트로피.(범주형 손실함수를 사용함)
    # 신경망이 훈련하는 동안 정확도를 리포트하려고 한다.
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # 학습시작
    print("start to fit...")
    model.fit(tr_img, tr_label, epochs=epoch)

    # 테스트
    # print("start to evaluate...")
    model.evaluate(test_img, test_label)

    # Generates output predictions for the input samples.
    classification_list = model.predict(test_img)
    print(classification_list[0])
    print(test_label[0])

class CallBackFunc(tf.keras.callbacks.Callback):
    """
    이 클래스는 tf.keras.callbacks.Callback 클래스를 상속한다.
    """
    def __init__(self, acc):
        super().__init__()
        self.acc = acc

    def on_epoch_begin(self, epoch, logs=None):
        # tf.keras.callbacks.Callback 클래스의 함수를 상속한 듯하다.
        print('훈련을 시작합니다..')

    def on_epoch_end(self, epoch, logs=None):
        # logs 에는 에폭에 대한 정보가 있고, 정확도를 꺼낼 수 있다.
        if logs.get('accuracy') > self.acc:
            print(f'\n정확도 {self.acc*100}% 에 도달하여 훈련을 종료합니다.....')

            # 훈련을 멈추는 flag 이다.
            self.model.stop_training = True

            print(logs)
            # logs 는 dictionary 이다.
            # {'loss': 0.3763212561607361, 'accuracy': 0.8641999959945679}

def training_using_callback( epoch, acc ):
    """
    정확도를 체크하는 callback 을 이용하여 훈련 조기 종료시키기
    :param acc:
    :param epoch:
    :return:
    """
    callback = CallBackFunc( acc )

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # 학습시작하는데 callback 를 이용하겠다.
    # 그 객체로 tf.keras.callbacks.Callback 을 상속한 클래스 인스턴스를 사용하겠다.
    model.fit(tr_img, tr_label, epochs=50, callbacks=[callback])

if __name__ == '__main__' :

    # epoch 5
    # training( epoch=5 )
    # 훈련 정확도 0.89, 테스트 평가 정확도 0.87

    # 에폭을 늘린다 5 -> 50
    training( epoch=50 )
    # 훈련 정확도 0.96, 테스트 평가 정확도 0.89
    # 5회에서 50으로 epoch 을 늘렸을 때 훈련정확도는 89% -> 96% 로 많이 증가되었는데
    # 테스트 정확도는 87% -> 89% 로 많이 향상되지는 않았다.

    # 해석 :
    # 훈련세트에 특화되었다고 볼 수 있다. 과대적합.

    # 모델이 훈련 시 정확도 n% 에 도달할 때까지 훈련하고 싶을 때는 콜백을 이용하면 좋다.
    # training_using_callback(epoch=50, acc=0.90)




