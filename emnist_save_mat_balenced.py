from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
from numpy import argmax
from scipy.io import loadmat


# 1. 데이터셋 생성하기

# EMNIST byclass 데이터셋 로드 (0-9, A-Z, a-z 총 62클래스)
print("EMNIST 모델을 불러옵니다.")
mat_data = loadmat('matlab/emnist-byclass.mat')
x_train = mat_data['dataset'][0][0][0][0][0][0].reshape(-1, 28, 28)
y_train_original = mat_data['dataset'][0][0][0][0][0][1].flatten()  # 원본 라벨 보관
x_test = mat_data['dataset'][0][0][1][0][0][0].reshape(-1, 28, 28)
y_test = mat_data['dataset'][0][0][1][0][0][1].flatten()
print(f"훈련용 샘플: {len(x_train)}, 평가용 샘플: {len(x_test)}")

# 데이터셋 결측치 및 이상치 확인 
print("결측치 확인:")
print(f"x_train NaN 개수: {np.isnan(x_train).sum()}")
print(f"y_train NaN 개수: {np.isnan(y_train_original).sum()}")
print(f"x_test NaN 개수: {np.isnan(x_test).sum()}")
print(f"y_test NaN 개수: {np.isnan(y_test).sum()}")

print("픽셀값 범위:")
print(f"x_train 범위: {x_train.min()} ~ {x_train.max()}")
print(f"x_test 범위: {x_test.min()} ~ {x_test.max()}")

print("라벨 범위:")
print(f"y_train 범위: {y_train_original.min()} ~ {y_train_original.max()}")
print(f"y_test 범위: {y_test.min()} ~ {y_test.max()}")

print("데이터 형태:")
print(f"x_train shape: {x_train.shape}")
print(f"y_train 고유값 개수: {len(np.unique(y_train_original))}")
print("=" * 25 + "\n")

# 데이터셋 전처리
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
num_classes = 62  # 0-9(10) + A-Z(26) + a-z(26) = 62
y_train = np_utils.to_categorical(y_train_original, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# 클래스 가중치 계산 (불균형 해결)
class_weight_dict = {}
for i in range(62):
    if i < 10:      # 숫자
        class_weight_dict[i] = 0.9
    elif i < 36:    # 대문자  
        class_weight_dict[i] = 1.1
    else:           # 소문자
        class_weight_dict[i] = 1.3

# 훈련용 데이터에서 훈련셋과 검증셋(학습 중 성능 모니터링용) 분리 (30% 검증셋)
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

# 4. 모델 학습시키기 (클래스 가중치 적용)
print("클래스 가중치를 적용하여 모델 학습을 시작합니다.")
model.fit(x_train, y_train, 
          epochs=10,  # epochs 는 반복학습 횟수 이다. 현재로서는 10번이 적절, 그 이상은 과적합 우려와 자원, 메모리 문제등이 생길 수 있음
          batch_size=128, 
          validation_data=(x_val, y_val),
          class_weight=class_weight_dict,  # 클래스 불균형 해결
          verbose=1)

# 5. 모델 평가하기
print("모델을 평가합니다.")
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(f'테스트 손실 : {loss_and_metrics[0]:.4f}')
print(f'테스트 정확도 : {loss_and_metrics[1]:.4f}')

# 6. 모델 저장하기
model.summary()
model.save("emnist_model_balanced_2.h5")
print("모델이 .h5 확장자로 해당 .py 파일과 같은 경로에 저장되었습니다.")