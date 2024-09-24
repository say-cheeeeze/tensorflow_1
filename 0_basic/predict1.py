import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# tensorflow 를 사용할 때에는 Sequential 클래스로 층을 정의한다.
# Sequential 클래스 안에서 층의 형태를 지정한다.
# keras.layers API 를 사용해 층을 정의할 수 있다.
# Dense 는 뉴런이 완전히 조밀하게 연결된 것을 의미함
# units=1 이므로 1개의 뉴런을 가지는 층이다.
# 신경망의 첫번째 층 입력 데이터 크기를 지정한다. 숫자 한개 이므로 1 로 지정
model = Sequential( [Dense(units=1, input_shape=[1])])

# 사실 y = 2x-1 . 하지만 관계를 모르기 때문에 추측을 시작한다.
# 손실함수 loss function 는 추측을 측정한다.
# 손실 함수가 반환한 정보를 가지고 (정답과 차이) 다시 추측을 시작한다.
# 이를 담당하는 것이 옵티마이저 optimizer 이다.
# 이 때 미적분이 많이 사용되지만 텐서플로를 사용하면 이 과정을 감출 수 있다.
# 상황에 따라 적절한 옵티마이저를 고르기만 하면 된다.
# 지금은 sgd 라는 옵티마이저를 사용했다. ( 확률적 경사 하강법 stochastic gradient descent 의 약자이다 )
# 추측한 값에서 오차를 계산한 결과가 주어졌을 때 또 다른 추측을 만드는 복잡한 수학 함수이다.
# 이 과정을 반복해 손실을 최소화하며 이를 통해 추측이 정답에 점점 더 가깝도록 만든다.
model.compile( optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 훈련을 시작하는 명령어 xs 와 ys 를 사용하고 500번 반복해라.
# 첫번쨰 반복에서 이 추측이 얼마나 잘 됐는지 측정한다. 이 결과를 옵티마이저에 피드백하고 새로운 추측을 생성한다.
# 손실(또는 오차)가 시간이 지남에 따라 줄어드는 로직을 사용해 이 과정을 반복한다
# 결과적으로 추측은 점점 더 좋아진다.
model.fit(xs, ys, epochs=500)

# x 가 10.0 일 때 Y 를 예측하면?
print( 'x 가 10 일때 : ', model.predict([10.0]))
print( 'x 가 24 일때 : ', model.predict([24.0]))

