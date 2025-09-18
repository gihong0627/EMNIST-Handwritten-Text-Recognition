import numpy as np
from keras.models import load_model
from scipy.io import loadmat
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# made in claude ai
# 특정 테스트셋을 두가지 모델에 집어넣어 모델의 성능을 직접적으로 비교하는 코드
# 각 클래스에 그룹에 가해질 가중치를 수동으로 조정하는 과정에서 현재 모듈의 상태를 확인하기 위해 제작 의뢰
# 결과를 시각화하고 분석하는 기능 포함 


# 테스트 데이터 로드
print("테스트 데이터 로딩 중")
mat_data = loadmat('matlab/emnist-byclass.mat')   # 테스트셋을 가지는 데이터셋 경로 설정                             
x_test = mat_data['dataset'][0][0][1][0][0][0].reshape(-1, 28, 28)
y_test_original = mat_data['dataset'][0][0][1][0][0][1].flatten()                    

# 전처리
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255.0
y_test = np_utils.to_categorical(y_test_original, 62)

print(f"테스트 샘플: {len(x_test)}")

# 모델 로드               
print("모델들을 불러오는 중")
try:
    model_original = load_model('emnist_model_2.h5')   # 비교의 기준이 될 모델을 선택 
    print("기존 모델 (emnist_model_2.h5) 로드 완료")
    has_original = True
except:
    print("기존 모델을 찾을 수 없습니다")
    has_original = False

try:
    model_balanced = load_model('emnist_model_balanced_2.h5') # 비교를 할 균형 모델(가중치 적용 모델)
    print("균형 모델 (emnist_model_balanced_2.h5) 로드 완료")
    has_balanced = True
except:
    print("균형 모델을 찾을 수 없습니다")
    has_balanced = False

if not has_original and not has_balanced:
    print("비교할 모델이 없습니다!")
    exit()

# 클래스별 성능 계산 함수
def calculate_class_performance(y_true, y_pred):
    digits_mask = y_true < 10
    upper_mask = (y_true >= 10) & (y_true < 36)
    lower_mask = y_true >= 36
    
    digits_acc = np.mean(y_pred[digits_mask] == y_true[digits_mask]) if np.sum(digits_mask) > 0 else 0
    upper_acc = np.mean(y_pred[upper_mask] == y_true[upper_mask]) if np.sum(upper_mask) > 0 else 0
    lower_acc = np.mean(y_pred[lower_mask] == y_true[lower_mask]) if np.sum(lower_mask) > 0 else 0
    
    return digits_acc, upper_acc, lower_acc

# 결과 저장용
results = {}

# 기존 모델 평가
if has_original:
    print("\n=== 기존 모델 평가 ===")
    loss_orig, acc_orig = model_original.evaluate(x_test, y_test, verbose=0)
    y_pred_orig = model_original.predict(x_test, verbose=0)
    y_pred_orig_labels = np.argmax(y_pred_orig, axis=1)
    
    digits_acc, upper_acc, lower_acc = calculate_class_performance(y_test_original, y_pred_orig_labels)
    
    results['original'] = {
        'overall_acc': acc_orig,
        'digits_acc': digits_acc,
        'upper_acc': upper_acc,
        'lower_acc': lower_acc,
        'loss': loss_orig
    }
    
    print(f"전체 정확도: {acc_orig:.4f}")
    print(f"숫자(0-9) 정확도: {digits_acc:.4f}")
    print(f"대문자(A-Z) 정확도: {upper_acc:.4f}")
    print(f"소문자(a-z) 정확도: {lower_acc:.4f}")
    print(f"손실: {loss_orig:.4f}")

# 균형 모델 평가
if has_balanced:
    print("\n=== 균형 모델 평가 ===")
    loss_bal, acc_bal = model_balanced.evaluate(x_test, y_test, verbose=0)
    y_pred_bal = model_balanced.predict(x_test, verbose=0)
    y_pred_bal_labels = np.argmax(y_pred_bal, axis=1)
    
    digits_acc, upper_acc, lower_acc = calculate_class_performance(y_test_original, y_pred_bal_labels)
    
    results['balanced'] = {
        'overall_acc': acc_bal,
        'digits_acc': digits_acc,
        'upper_acc': upper_acc,
        'lower_acc': lower_acc,
        'loss': loss_bal
    }
    
    print(f"전체 정확도: {acc_bal:.4f}")
    print(f"숫자(0-9) 정확도: {digits_acc:.4f}")
    print(f"대문자(A-Z) 정확도: {upper_acc:.4f}")
    print(f"소문자(a-z) 정확도: {lower_acc:.4f}")
    print(f"손실: {loss_bal:.4f}")

# 비교 시각화
if has_original and has_balanced:
    print("\n=== 성능 비교 시각화 ===")
    
    # 1. 클래스별 정확도 비교
    categories = ['Digits\n(0-9)', 'Upper\n(A-Z)', 'Lower\n(a-z)', 'Overall']
    original_scores = [results['original']['digits_acc'], 
                      results['original']['upper_acc'], 
                      results['original']['lower_acc'],
                      results['original']['overall_acc']]
    balanced_scores = [results['balanced']['digits_acc'], 
                      results['balanced']['upper_acc'], 
                      results['balanced']['lower_acc'],
                      results['balanced']['overall_acc']]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x_pos - width/2, original_scores, width, label='Original Model', alpha=0.8, color='lightcoral')
    plt.bar(x_pos + width/2, balanced_scores, width, label='Balanced Model', alpha=0.8, color='lightblue')
    
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x_pos, categories)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for i, (orig, bal) in enumerate(zip(original_scores, balanced_scores)):
        plt.text(i - width/2, orig + 0.02, f'{orig:.3f}', ha='center', va='bottom', fontsize=10)
        plt.text(i + width/2, bal + 0.02, f'{bal:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 2. 개선/악화 분석
    print("\n=== 성능 변화 분석 ===")
    print("균형 모델 vs 기존 모델:")
    
    improvements = {
        '전체 정확도': results['balanced']['overall_acc'] - results['original']['overall_acc'],
        '숫자 정확도': results['balanced']['digits_acc'] - results['original']['digits_acc'],
        '대문자 정확도': results['balanced']['upper_acc'] - results['original']['upper_acc'],
        '소문자 정확도': results['balanced']['lower_acc'] - results['original']['lower_acc']
    }
    
    for metric, improvement in improvements.items():
        status = "향상" if improvement > 0 else "하락" if improvement < 0 else "동일"
        print(f"{metric}: {improvement:+.4f} ({status})")
    
    # 3. 요약
    print(f"\n=== 요약 ===")
    avg_char_acc_orig = (results['original']['upper_acc'] + results['original']['lower_acc']) / 2
    avg_char_acc_bal = (results['balanced']['upper_acc'] + results['balanced']['lower_acc']) / 2
    
    print(f"문자 인식 개선: {avg_char_acc_bal - avg_char_acc_orig:+.4f}")
    print(f"숫자 인식 변화: {improvements['숫자 정확도']:+.4f}")
    
    if avg_char_acc_bal > avg_char_acc_orig and improvements['숫자 정확도'] >= -0.05:
        print("클래스 균형 조정이 성공적입니다")
    elif avg_char_acc_bal > avg_char_acc_orig:
        print("문자 인식은 개선되었으나 숫자 인식이 크게 하락했습니다.")
    else:
        print("클래스 균형 조정 효과가 미미합니다.")

elif has_original:
    print("\n균형 모델이 없어 비교할 수 없습니다. 기존 모델 성능만 표시됩니다.")
elif has_balanced:
    print("\n기존 모델이 없어 비교할 수 없습니다. 균형 모델 성능만 표시됩니다.")

print("\n평가 완료!")