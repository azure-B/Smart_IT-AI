from config import config
from sentence_transformers import SentenceTransformer, util
from read_data import extract_speaker_text_from_json_in_folder
import torch
import os
import json

# ==========================================
# 1. 장치 자동 설정 (가장 중요한 부분)
# ==========================================
# CUDA가 있으면 'cuda', 없으면 'cpu'로 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

print("-" * 30)
if device == "cuda":
    print(f"CUDA. ({torch.cuda.get_device_name(0)})")
else:
    print("CPU")
print("-" * 30)

EMBEDDING_FILE = config.EMBEDDING_FILE
TEXT_DATA_FILE = config.TEXT_DATA_FILE

# 2. 모델 로드 (감지된 장치에 맞게 로드)
model = SentenceTransformer('jhgan/ko-sbert-nli', device=device)

if os.path.exists(EMBEDDING_FILE) and os.path.exists(TEXT_DATA_FILE):
    print("--- 저장된 데이터를 로드합니다 ---")
    dataset_embeddings = torch.load(EMBEDDING_FILE, map_location=device)
    with open(TEXT_DATA_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

else:
    print("--- 데이터셋 생성 및 임베딩 시작 ---")
    test_path = os.path.join("dataset", "Training")
    dataset = extract_speaker_text_from_json_in_folder(test_path)

    if not dataset:
        print("오류: 데이터셋을 찾을 수 없습니다.")
        exit()
    dataset_embeddings = model.encode(dataset, convert_to_tensor=True)

    torch.save(dataset_embeddings, EMBEDDING_FILE)
    with open(TEXT_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"--- 데이터 준비 완료 (총 {len(dataset)}개) ---")
print("-" * 30)

while True:
    try:
        user_speech = input(" 입력 (종료하려면 '종료'): ")
    except EOFError:
        break

    if user_speech.strip():
        if "종료" in user_speech.replace(" ", ""):
            print(" 프로그램 종료 ")
            break

        user_speech_embedding = model.encode(user_speech, convert_to_tensor=True)
        hits = util.semantic_search(user_speech_embedding, dataset_embeddings, top_k=3)

        print(f"\n[입력]: {user_speech}")

        for hit in hits[0]:
            matched_idx = hit['corpus_id']
            similarity_score = hit['score']
            matched_text = dataset[matched_idx]

            if "답변:" in matched_text:
                answer_only = matched_text.split("답변:", 1)[1].strip()
            else:
                answer_only = matched_text

            print(answer_only)

    else:
        print("내용을 입력해주세요.")

    print("-" * 30)