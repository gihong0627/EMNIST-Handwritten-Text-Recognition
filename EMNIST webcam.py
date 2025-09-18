import numpy as np
import cv2
from keras.models import load_model
import time
current_prediction = "대기중"
current_confidence = 0.0

print("EMNIST 모델 불러오는중")
model = load_model('emnist_model_balanced_2.h5')
print("EMNIST 모델 불러오기 완료.")

# EMNIST 클래스 매핑
def get_class_name(class_index):
    if class_index < 10:
        return str(class_index)  # 0~9
    elif class_index < 36:
        return chr(ord('A') + class_index - 10)  # A~Z
    else:
        return chr(ord('a') + class_index - 36)  # a~z

def preprocess_for_emnist(roi): #ROI(캠 중앙의 픽셀영역)를 EMNIST 형식으로 전처리
    # 28x28로 리사이즈
    resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # EMNIST 변환
    transformed = np.rot90(resized, k=3)
    transformed = np.fliplr(transformed)
    
    # 정규화
    normalized = transformed.astype('float32') / 255.0
    
    return normalized.reshape(1, 784)

def main():  
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다, 웹캠 환경을 준비해주세요.")
        return
    
    print("웹캠을 열어주세요.")
    print("\n==== 조작법 ====")
    print("웹캠 중앙의 녹색 박스 안에 문자를 보여주세요.")
    print("준비가 되면 스페이스바를 눌러 예측을 시작합니다.")
    print("'r' 키를 눌러 ROI 박스 표시를 off할 수 있습니다.")
    print("제출 - 'space'키를 눌러 예측 시작")
    print("종료 - 'x'키를 눌러 웹캠 종료")
    

    show_roi = True
    last_prediction_time = 0
    prediction_cooldown = 1.0  # 1초 쿨다운
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 뒤집기 (거울 효과, 화면 표시용)
        frame = cv2.flip(frame, 1)
        
        # ROI(캠 중앙의 픽셀영역) 영역 설정 (중앙 200x200)
        h, w = frame.shape[:2]
        roi_size = 200
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # ROI 추출
        roi = frame[y1:y2, x1:x2]
        
        # ROI를 그레이스케일로 변환
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 이진화
        _, roi_binary = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # ROI 박스 그리기
        if show_roi:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "ROI", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 자동 예측 (제출 없이도 주기적으로 현재 roi에 인식되고 있는 문자를 예측)
        current_time = time.time()
        if current_time - last_prediction_time > prediction_cooldown:
            # ROI가 충분히 내용이 있는지 확인
            if np.sum(roi_binary) > 1000:  # 임계값 조정 가능
                processed = preprocess_for_emnist(roi_binary)
                prediction = model.predict(processed, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                if confidence >= 40: # 신뢰도가 40% 이상일 때만 갱신
                    predicted_char = get_class_name(class_idx)

                    current_prediction = predicted_char
                    current_confidence = confidence

                last_prediction_time = current_time

        cv2.putText(frame, f"prediction: {current_prediction}", 
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"confidence: {current_confidence:.1f}%", 
           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 현재 처리된 ROI 표시 (화면 오른쪽 위 작은 창)
        roi_display = cv2.resize(roi_binary, (100, 100), interpolation=cv2.INTER_NEAREST)
        frame[10:110, w-110:w-10] = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "Processed", (w-110, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 화면 표시
        cv2.imshow('EMNIST webcam', frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('x'):
            break
        elif key == ord('r'):
            show_roi = not show_roi
        elif key == ord(' '): #  수동 예측
            processed = preprocess_for_emnist(roi_binary)
            prediction = model.predict(processed, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_char = get_class_name(class_idx)
            
            print(f"\n제출 내용 예측: '{predicted_char}' ({confidence:.1f}%)")
            
            # 상위 3개 결과
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            for i, idx in enumerate(top3_indices):
                char = get_class_name(idx)
                conf = prediction[0][idx] * 100
                print(f"  {i+1}. '{char}' ({conf:.1f}%)")
    
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료.")

if __name__ == "__main__":  # 프로그램 실행 
    main()