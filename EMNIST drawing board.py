import numpy as np
import cv2
from keras.models import load_model

# EMNIST 클래스 매핑 (0-9, A-Z, a-z)
def get_class_name(class_index):
    if class_index < 10:
        return str(class_index)  # 0-9
    elif class_index < 36:
        return chr(ord('A') + class_index - 10)  # A-Z
    else:
        return chr(ord('a') + class_index - 36)  # a-z

print("EMNIST 모델 불러오는중")
model = load_model('emnist_model_balanced_2.h5')
print("EMNIST 모델 불러오기 완료.")

onDown = False
xprev, yprev = None, None

def onmouse(event, x, y, flags, params):
    global onDown, img, xprev, yprev
    if event == cv2.EVENT_LBUTTONDOWN:
        onDown = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if onDown == True:
            cv2.line(img, (xprev,yprev), (x,y), (255,255,255), 15)   # 15는 GUI 선 굵기, 15~ 20사이가 가장 적당함. 
    elif event == cv2.EVENT_LBUTTONUP:
        onDown = False
    xprev, yprev = x,y

def preprocess_image(img):  #완성 이미지 제출 전 모델에 적합한 기본적인 전처리 
    # 28x28로 리사이즈
    x_resize = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
    
    # 그레이스케일 변환
    x_gray = cv2.cvtColor(x_resize, cv2.COLOR_BGR2GRAY)
    
    # EMNIST 데이터셋에 맞춘 변환 (모델의 이미지는 기본적으로 90도 회전, 좌우반전된 상태로 저장되어있음.)
    x_gray = np.rot90(x_gray, k=3)  # 270도 회전 (시계 반대 방향으로 90도)
    x_gray = np.fliplr(x_gray)      # 좌우 반전
    
    # 정규화 (0-255 -> 0-1)
    x_normalized = x_gray.astype('float32') / 255.0
    
    # 모델 입력 형태로 변환 (1, 784)
    x_flattened = x_normalized.reshape(1, 28*28)
    
    # 손글씨 제출하면 디버깅용 전처리된 이미지 확인 (GUI 옆에(아마 왼쪽) 표시)
    debug_img = (x_normalized * 255).astype(np.uint8)
    debug_img_large = cv2.resize(debug_img, (140, 140), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("preprocessed image)", debug_img_large)
    
    return x_flattened

cv2.namedWindow("EMNIST drawing board")
cv2.namedWindow("preprocessed image")
cv2.setMouseCallback("EMNIST drawing board", onmouse)

width, height = 280, 280
img = np.zeros((280,280,3), np.uint8)

print("\n=== EMNIST 손글씨 그림판 ===")
print("조작법:")
print("마우스 좌클릭으로 그리기")
print(" 지우개 - 'r'키를 눌러 reset")
print(" 제출 - 's'키를 눌러 submit")
print(" 보정 - 'c'키를 눌러 회전, 반전, 둘다 적용")
print(" 종료 - 'x'키를 눌러 종료")
print("="*30 + "\n")

correction_mode = 0  # 보정모드 0: 기본(회전+반전), 1: 회전, 2: 반전, 3: 둘 다 미적용

while True:
    cv2.imshow("EMNIST drawing board", img)
    key = cv2.waitKey(1)
    
    if key == ord('r'):
        img = np.zeros((280,280,3), np.uint8)
        print("지우기 완료.")
        
    if key == ord('t'):
        correction_mode = (correction_mode + 1) % 4
        modes = ["기본(회전+반전)", "회전", "반전", "둘 다 미적용"]
        print(f"보정 모드: {modes[correction_mode]}")
        
    if key == ord('s'):
        # 이미지 전처리
        x_resize = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
        x_gray = cv2.cvtColor(x_resize, cv2.COLOR_BGR2GRAY)
        
        # 보정 모드에 따른 처리
        if correction_mode == 0:  # 기본 (회전 + 반전)
            x_gray = np.rot90(x_gray, k=3)
            x_gray = np.fliplr(x_gray)
        elif correction_mode == 1:  # 회전만
            x_gray = np.rot90(x_gray, k=3)
        elif correction_mode == 2:  # 반전만
            x_gray = np.fliplr(x_gray)
        # correction_mode == 3은 보정 없음
        
        # 완성 이미지 정규화 및 형태 변환
        x_normalized = x_gray.astype('float32') / 255.0
        x = x_normalized.reshape(1, 28*28)
        
        # 디버깅용 전처리 이미지 표시
        debug_img = (x_normalized * 255).astype(np.uint8)
        debug_img_large = cv2.resize(debug_img, (140, 140), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("preprocessed image", debug_img_large) 

        # 모델 예측
        y_pred = model.predict(x, verbose=0)
        y_class = np.argmax(y_pred, axis=1)[0]
        confidence = np.max(y_pred) * 100
        
        predicted_char = get_class_name(y_class)
        print(f"Prediction: '{predicted_char}' (Confidence: {confidence:.1f}%)")
        
        # 상위 5개 예측 결과 표시
        top5_indices = np.argsort(y_pred[0])[-5:][::-1]
        print("상위 5개의 정확도의 예측 결과:")
        for i, idx in enumerate(top5_indices):
            char = get_class_name(idx)
            conf = y_pred[0][idx] * 100
            print(f"  {i+1}. '{char}' ({conf:.1f}%)")
        print()
        cv2.waitKey(4000)  # 4초 대기
        cv2.destroyWindow("preprocessed image") # 디버깅용 전처리 이미지 닫기

    if key == ord('x'):
        print("프로그램 종료.")
        break

cv2.destroyAllWindows()