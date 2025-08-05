# ======================================================================
# 저장된 KoBERT 모델로 에세이 목적 분류하기 (추론용 - 수동 매핑)
# ======================================================================

# ───────────────────────────
# 1) 라이브러리 임포트 및 기본 설정
# ───────────────────────────
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 설정 ---
# 이전에 학습된 모델이 저장된 경로를 지정합니다.
MODEL_PATH = "./best_kobert_model"

# ───────────────────────────
# 2) GPU 설정
# ───────────────────────────
# 학습 시 사용했던 GPU 번호와 동일하게 맞추거나, 사용 가능한 GPU로 설정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ───────────────────────────
# 3) 저장된 모델 및 토크나이저 로드
# ───────────────────────────
print("\n--- 저장된 모델과 토크나이저를 불러옵니다... ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    print("로드 완료.")
except OSError:
    print(f"오류: '{MODEL_PATH}' 폴더를 찾을 수 없거나, 폴더 안에 모델 파일이 없습니다.")
    print("학습 코드를 먼저 실행하여 모델을 저장해주세요.")
    exit()

# [수정] model.config.id2label 대신, 수동으로 라벨 맵을 정의합니다.
# [주의!] 이 순서는 반드시 학습 코드 실행 시 출력되었던 `레이블 매핑`과 동일해야 합니다.
id2label = {0: '설득', 1: '설명', 2: '친교 및 정서'}
print(f"사용자 정의 레이블 매핑: {id2label}")

# ───────────────────────────
# 4) 예측 함수 정의
# ───────────────────────────
def classify_essay_purpose(text: str):
    """
    입력된 텍스트의 목적을 분류하고, 각 클래스별 확률을 반환합니다.
    """
    model.eval()

    inputs = tokenizer(
        text, 
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    
    prediction_idx = torch.argmax(probs, dim=-1).item()
    
    # 위에서 직접 정의한 id2label 딕셔너리를 사용하여 라벨 이름으로 변환합니다.
    predicted_label = id2label[prediction_idx]
    
    class_probabilities = {id2label[i]: prob.item() for i, prob in enumerate(probs[0])}
    
    return predicted_label, class_probabilities

# ───────────────────────────
# 5) 새로운 데이터로 테스트 실행
# ───────────────────────────
test_sentence_1 = "지구 온난화는 전 지구적 기온 상승을 의미하며, 이는 산업 혁명 이후 화석 연료 사용의 급증으로 인한 온실가스 농도 증가가 주된 원인으로 지목됩니다. 이 현상은 해수면 상승, 생물 다양성 감소, 그리고 가뭄 및 홍수와 같은 극단적 기상 이변의 빈도 증가를 초래하여 인류의 생존 환경에 직접적인 위협이 되고 있습니다."
test_sentence_2 = "일회용품 사용을 줄이는 것은 더 이상 선택이 아닌 필수입니다. 무분별한 플라스틱 소비는 해양 생태계를 파괴하고 결국 우리의 건강까지 위협하고 있습니다. 따라서 개인의 작은 불편함을 감수하고 텀블러 사용과 장바구니 이용을 생활화함으로써, 우리는 미래 세대를 위한 지속 가능한 환경을 만들어갈 책임이 있습니다."
test_sentence_3 = "어제저녁, 창밖으로 보이던 노을이 유난히 아름다워서 한참을 넋 놓고 바라봤어. 하루 종일 복잡했던 마음이 스르르 녹아내리는 기분이었지. 가끔은 이렇게 아무 생각 없이 하늘을 보는 것만으로도 큰 위로가 되는 것 같아. 너도 오늘 하루 힘들었다면, 잠시 하늘 한번 올려다봤으면 좋겠다."

test_texts = [test_sentence_1, test_sentence_2, test_sentence_3]

print("\n--- 새로운 텍스트 분류 테스트 시작 ---")
for i, text in enumerate(test_texts):
    predicted_class, probabilities = classify_essay_purpose(text)
    print(f"\n[ 문장 {i+1} ]")
    print(f"입력 텍스트: {text}")
    print(f"▶ 예측 결과: {predicted_class}")
    print(f"▶ 클래스별 확률:")
    # 확률을 내림차순으로 정렬하여 보기 좋게 출력
    for label, prob in probabilities.items():
        print(f"  - {label}: {prob:.2%}")