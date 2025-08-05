# =================================================================================
# KoBERT 기반 에세이 목적 분류기 (Hugging Face Trainer 최신 버전)
# =================================================================================

# ───────────────────────────
# 0) 기본 세팅 & 라이브러리 임포트
# ───────────────────────────
import pandas as pd
import torch
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler # 데이터 불균형 처리를 위해 추가
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- 하이퍼파라미터 ---
MODEL_NAME = "klue/bert-base"
TRAIN_PATH = "./train_label.csv"  # 학습 데이터 경로
VALID_PATH = "./valid_label.csv"  # 검증 데이터 경로
MAX_LEN = 512
BATCH_SIZE = 16  
LR = 1e-5
EPOCHS = 5
SEED = 42

# ───────────────────────────
# 1) 재현성을 위한 시드 고정
# ───────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ───────────────────────────
# 2) 데이터 로드 및 전처리
# ───────────────────────────
def load_and_preprocess_data(file_path):
    # CSV의 첫 번째 열은 'essay_text', 두 번째 열은 'essay_purpose'라고 가정
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8')
    
    df.dropna(subset=['essay_text', 'essay_purpose'], inplace=True)
    df['essay_text'] = df['essay_text'].str.replace(r'#@\w+#', '', regex=True).str.strip()
    df = df[df['essay_text'].str.len() > 0]
    return df

print("--- 데이터 로드 시작 ---")
train_df = load_and_preprocess_data(TRAIN_PATH)
eval_df = load_and_preprocess_data(VALID_PATH)
print(f"원본 Train 데이터셋 크기: {len(train_df)}")
print(f"원본 Valid 데이터셋 크기: {len(eval_df)}")

print("--- 학습(Train) 데이터셋의 'essay_purpose' 컬럼 확인 ---")
print(train_df['essay_purpose'].value_counts())

print("\n--- 검증(Valid) 데이터셋의 'essay_purpose' 컬럼 확인 ---")
print(eval_df['essay_purpose'].value_counts())

# ───────────────────────────
# 3) 레이블 인코딩 및 오버샘플링 (데이터 불균형 해결)
# ───────────────────────────
print("\n--- 레이블 인코딩 및 오버샘플링 시작 ---")
label_encoder = LabelEncoder()
label_encoder.fit(train_df['essay_purpose'])
train_df['labels'] = label_encoder.transform(train_df['essay_purpose'])
eval_df['labels'] = label_encoder.transform(eval_df['essay_purpose'])

label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
print(f"레이블 매핑: {label_mapping}")

print("\n오버샘플링 전 학습 데이터 분포:")
print(train_df['labels'].value_counts().sort_index())

ros = RandomOverSampler(random_state=SEED)
X_resampled, y_resampled = ros.fit_resample(train_df[['essay_text']], train_df['labels'])
train_df_balanced = pd.DataFrame(X_resampled, columns=['essay_text'])
train_df_balanced['labels'] = y_resampled

print("\n오버샘플링 후 학습 데이터 분포:")
print(train_df_balanced['labels'].value_counts().sort_index())

# ───────────────────────────
# 4) Hugging Face Dataset 변환 및 토큰화
# ───────────────────────────

print("\n--- Dataset 변환 및 토큰화 시작 ---")
train_dataset = Dataset.from_pandas(train_df_balanced)
eval_dataset = Dataset.from_pandas(eval_df[['essay_text', 'labels']])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples['essay_text'], truncation=True, max_length=MAX_LEN)

# .map()을 사용하여 토큰화 및 'essay_text' 컬럼 제거
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['essay_text'])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['essay_text'])

if 'token_type_ids' in tokenized_train_dataset.column_names:
    print("\n[수정] 문제가 되는 'token_type_ids' 컬럼을 데이터셋에서 제거합니다.")
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['token_type_ids'])
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(['token_type_ids'])

# --- 최종 피처 확인 ---
print("\n--- 최종 데이터 피처 확인 ---")
print("최종 학습 데이터셋 컬럼:", tokenized_train_dataset.column_names)
print("최종 검증 데이터셋 컬럼:", tokenized_eval_dataset.column_names)

# 첫 번째 학습 데이터 샘플을 통해 실제 데이터 구조 확인
print("\n첫 번째 학습 데이터 샘플:")
print(tokenized_train_dataset[0])


# ───────────────────────────
# 5) 모델 및 Trainer 설정
# ───────────────────────────
print("\n--- 모델 및 Trainer 설정 시작 ---")
print(len(label_mapping))
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_mapping))

training_args = TrainingArguments(
    output_dir="./results_balanced",
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_balanced",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True,  # 학습 속도 향상을 위한 혼합 정밀도 사용
    seed=SEED,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    return {"accuracy": accuracy, "macro_f1": f1, "precision": precision, "recall": recall}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# ───────────────────────────
# 6) 모델 학습 및 최종 평가
# ───────────────────────────
print("\n--- 최종 모델 학습 시작 ---")
trainer.train()

BEST_MODEL_PATH = "./best_kobert_model" # 베스트 모델을 저장할 새 폴더 경로
print(f"\n--- 가장 성능이 좋은 모델을 '{BEST_MODEL_PATH}'에 저장합니다 ---")
trainer.save_model(BEST_MODEL_PATH)
print("저장 완료.")

# 2. 토크나이저도 같은 곳에 저장 
tokenizer.save_pretrained(BEST_MODEL_PATH)



print("\n--- 최종 평가 ---")
# trainer.evaluate()는 현재 trainer.model에 로드된 베스트 모델로 평가를 진행합니다.
final_results = trainer.evaluate()
print(final_results)