import numpy as np
import cv2
from keras.models import load_model
import os
import glob
import re
import pytesseract
from datetime import datetime
import sys


pytesseract.pytesseract.tesseract_cmd = 'D:/Tesseract-OCR/tesseract.exe'  #실제 pc의 테서렉트 경로를 설정.

def extract_characters_contours(image_path, output_folder):  # 윤곽선으로 문자 분리 및 저장
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return []
    
    print(f"윤곽선 방법으로 문자 추출 중")
    
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 모폴로지 연산
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 윤곽선을 x 좌표 순으로 정렬
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    saved_files = []
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # 너무 작은 윤곽선 무시
        if w < 10 or h < 10:
            continue
        
        # 문자 영역 추출
        margin = 8
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        char_image = original[y1:y2, x1:x2]
        
        # 파일명 생성 및 저장
        filename = f"contour_char_{i+1:03d}.png"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, char_image)
        saved_files.append(filename)
        print(f" 저장: {filename} (크기: {char_image.shape[1]}x{char_image.shape[0]})")
    
    return saved_files

def extract_characters_tesseract(image_path, output_folder):  # Tesseract로 문자 위치 찾아 분리 및 저장
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return []
    
    print(f"Tesseract 방법으로 문자 추출 중")
    
    # Tesseract로 문자 위치 정보 얻기
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config='--psm 6')
    except Exception as e:
        print(f"Tesseract 오류: {e}")
        return []
    
    saved_files = []
    char_count = 0
    
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:
            text = data['text'][i].strip()
            if text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # 각 문자별로 분리
                for j, char in enumerate(text):
                    if char.strip():
                        char_count += 1
                        
                        # 문자별 위치 추정
                        char_w = w // len(text)
                        char_x = x + j * char_w
                        
                        # 안전한 범위 확인
                        if (char_x >= 0 and char_x + char_w <= image.shape[1] and 
                            y >= 0 and y + h <= image.shape[0]):
                            
                            char_image = image[y:y+h, char_x:char_x+char_w]
                            
                            # 파일명 생성 (인식된 문자 포함)
                            safe_char = char if char.isalnum() else f"special_{ord(char)}"
                            filename = f"tesseract_char_{char_count:03d}_{safe_char}.png"
                            filepath = os.path.join(output_folder, filename)
                            cv2.imwrite(filepath, char_image)
                            saved_files.append(filename)
                            
                            print(f"  저장: {filename} (문자: '{char}', 크기: {char_image.shape[1]}x{char_image.shape[0]})")
    
    return saved_files

def create_output_folder(base_name):  # 출력 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_name}_{timestamp}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"출력 폴더 생성: {folder_name}")
    
    return folder_name

def extract_and_save_characters(image_path):  # 메인 추출 함수
    if not os.path.isfile(image_path):
        print(f"파일을 찾을 수 없습니다: {image_path}")
        return
    
    # 출력 폴더 생성
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = create_output_folder(base_name)
    
    print(f"\n이미지 처리 시작: {os.path.basename(image_path)}")
    print(f"출력 폴더: {output_folder}")
    print("-" * 60)
    
   
    extract_characters_tesseract(image_path, output_folder)
    extract_characters_contours(image_path, output_folder) #테서렉트 실패시 opencv만으로 이미지 추출 
    
    # 결과 요약
    print("-" * 60)
    print("이미지 문자별 추출 완료")
    print(f"{output_folder} 폴더에 저장되었습니다.")
    return output_folder

    
# EMNIST 클래스 매핑
def get_class_name(class_index):
    if class_index < 10:
        return str(class_index)  # 0-9
    elif class_index < 36:
        return chr(ord('A') + class_index - 10)  # A-Z
    else:
        return chr(ord('a') + class_index - 36)  # a-z

def preprocess_image(img_path): #추출한 이미지 파일을 EMNIST 형식으로 전처리
    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load image {img_path}")
        return None
    
    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 이진화 (배경은 검은색, 글자는 흰색으로)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 글자가 검은색인 경우 반전
    if np.mean(binary) > 127:  # 배경이 밝다면
        binary = 255 - binary
    
    # 28x28로 리사이즈
    resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
    
    # EMNIST 변환 (회전 + 반전)
    transformed = np.rot90(resized, k=3)
    transformed = np.fliplr(transformed)
    
    # 정규화
    normalized = transformed.astype('float32') / 255.0
    
    return normalized.reshape(1, 784)

def predict_single_image(model, img_path, show_debug=True):  #단일 이미지 파일 예측 
    processed = preprocess_image(img_path)
    if processed is None:
        return
    
    # 예측
    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    predicted_char = get_class_name(class_idx)
    
    print(f"\n이미지 파일 명: {os.path.basename(img_path)}")
    print(f"예측 결과: '{predicted_char}' (신뢰도: {confidence:.1f}%)\n")
    
    # 상위 3개 결과
    top3_indices = np.argsort(prediction[0])[-3:][::-1]
    print("상위 3개의 예측 결과:")
    for i, idx in enumerate(top3_indices):
        char = get_class_name(idx)
        conf = prediction[0][idx] * 100
        print(f"  {i+1}. '{char}' ({conf:.1f}%)")
    

def batch_predict(model, folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'] # 폴더 내 여러 확장자 이미지 일괄 처리 
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    image_files = list(set(image_files))

    if not image_files:
        print(f"{folder_path}폴더에 이미지 없음. 이미지 추출 실패")
        return
    
    results = []
    
    for img_path in image_files:
        processed = preprocess_image(img_path)
        if processed is not None:
            prediction = model.predict(processed, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_char = get_class_name(class_idx)
            
            results.append({
                'file': os.path.basename(img_path),
                'prediction': predicted_char,
                'confidence': confidence
            })
            
    def natural_sort_key(filename):  # 파일명에서 숫자를 추출해서 정수로 변환하여 정렬
        numbers = re.findall(r'\d+', filename)
        return [int(num) for num in numbers] if numbers else [0]
    results.sort(key=lambda x: natural_sort_key(x['file']))

    # 결과 출력
    print("\n" + "-"*45)
    print("이미지 예측 결과")
    print("-"*45 + "\n")
    for result in results:
        print(f"{result['file']:<20} → '{result['prediction']}' ({result['confidence']:.1f}%)")



def main():
    print("EMNIST 모델 불러오는 중")
    try:
        model = load_model('emnist_model.h5') # emnist_model_balanced_2.h5 모델과 전환해가며 테스트, 전체적인 성능은 emnist_model.h5이 아직 우수수
        print("EMNIST 모델 불러오기 성공")
    except:
        print("EMNIST 모델 불러오기 실패")
        return
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        if os.path.isfile(path):
            # 단일 문자 이미지 처리
            predict_single_image(model, path)
        elif os.path.isdir(path):
            # 여러문자 이미지 처리
            batch_predict(model, path)
        else:
            print(f"경로 확인 불가능: {path}")
    else:
        while True:
            print("1. 단일 문자 이미지 예측")
            print("2. 여러문자 이미지 추출 후 일괄 예측(ocr+opencv)")
            print("3. 종료")
            
            choice = input("\nChoose option (1-3): ").strip()
            
            if choice == '1':
                img_path = input("이미지 경로를 입력하세요: ").strip()
                if os.path.isfile(img_path):
                    predict_single_image(model, img_path)
                else:
                    print("이미지를 찾을 수 없습니다")
                    break
                    
            elif choice == '2':
                img_path = input("\n이미지 파일 경로를 입력하세요: ").strip()
                img_path = img_path.strip('"').strip("'")  # 따옴표 제거
    
                if not os.path.isfile(img_path):
                    print("파일을 찾을 수 없습니다.")
                    return
    
                extract_and_save_characters(img_path)

                folder_path = extract_and_save_characters(img_path)

                if os.path.isdir(folder_path):
                    batch_predict(model, folder_path)
                else:
                    print("폴더를 찾을 수 없습니다. 추출 기능에 문제가 발생했을 가능성이 높습니다.")
                    break
                    
            elif choice == '3':
                print("프로그램 종료.")
                break
            else:
                print("제대로된 선택을 해주세요.")

if __name__ == "__main__": # 프로그램 시작
    main()