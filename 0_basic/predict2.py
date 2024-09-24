import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

"""
모델 학습 후 가중치w와 절편b(편향치) 확인하기
학습된 모델의 Dense 에서 get_weights() 를 이용하여 확인할 수 있다.
"""

dense = Dense( units=1, input_shape=[1])
model = Sequential([dense])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0,-1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs,ys,epochs=500)

# x 값 하나의 출력값 예측
print( model.predict([10.0]))

# dense_weights() => 가중치와 절편 모두 반환.
print( f'신경망이 학습한 것. 1. dense.get_weights() : {dense.get_weights()}')
# model 객체의 layers 속성을 이용해 Dense 층 객체 조회할 수도 있다.
print( f'신경망이 학습한 것. 2. model.layers[0].get_weights() : {model.layers[0].get_weights()}')

dense_weight = dense.get_weights()
print(f'weight : {dense_weight[0]}')
print(f'bias : {dense_weight[1]}')
print( f'Y={dense_weight[0]}X + {dense_weight[1]}')
