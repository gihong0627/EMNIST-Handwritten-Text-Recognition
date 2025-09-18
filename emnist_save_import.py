# emnist_save.py
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
from numpy import argmax
from emnist import extract_training_samples, extract_test_samples

# 1. 데이터셋 생성하기

# EMNIST byclass 데이터셋 로드 (0-9, A-Z, a-z 총 62클래스)
print("EMNIST 모델을 불러옵니다.")  # 로컬 파일이 아닌 import 모듈로 불러오는 것.
x_train, y_train = extract_training_samples('byclass')
x_test, y_test = extract_test_samples('byclass')
print(f"훈련용 샘플: {len(x_train)}, 평가용 샘플: {len(x_test)}")

# 데이터셋 전처리
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
num_classes = 62  # 0~9(10) + A~Z(26) + a~z(26) = 62
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# 훈련셋과 검증셋 분리 (30% 검증셋)
val_split = int(len(x_train) * 0.3)
x_val = x_train[:val_split]
x_train = x_train[val_split:]
y_val = y_train[:val_split]
y_train = y_train[val_split:]

print(f"훈련셋: {len(x_train)}, 검증셋: {len(x_val)}")

# 2. 모델 구성하기 (더 복잡한 모델로 구성)
model = Sequential()
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
print("모델 학습을 시작합니다.")
model.fit(x_train, y_train, 
          epochs=10, 
          batch_size=128, 
          validation_data=(x_val, y_val),
          verbose=1)

# 5. 모델 평가하기
print("모델을 평가합니다.")
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(f'테스트 손실 : {loss_and_metrics[0]:.4f}')
print(f'테스트 정확도 : {loss_and_metrics[1]:.4f}')

# 6. 모델 저장하기
model.summary()
model.save("emnist_model.h5")
print("모델이 .h5 확장자로 해당 .py 파일과 같은 경로에 저장되었습니다.")