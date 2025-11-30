from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import os
import json
from sentence_transformers import SentenceTransformer, util

from config import config
from read_data import extract_speaker_text_from_json_in_folder
app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
dataset_embeddings = None
dataset = []

EMBEDDING_FILE = config.EMBEDDING_FILE
TEXT_DATA_FILE = config.TEXT_DATA_FILE

@app.on_event("startup")
async def startup_event():
    global model, dataset_embeddings, dataset
    print(f"서버 시작 : {device.upper()}")
    model = SentenceTransformer('jhgan/ko-sbert-nli', device=device)

    # 데이터 로드 또는 생성
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(TEXT_DATA_FILE):
        print("--- 저장된 데이터 로드 중 ---")
        # GPU에서 저장된 파일을 CPU에서 로드할 때 오류 방지를 위해 map_location 사용
        dataset_embeddings = torch.load(EMBEDDING_FILE, map_location=device)
        with open(TEXT_DATA_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        print("--- 데이터 생성 중 ---")
        test_path = os.path.join("dataset", "Training")
        dataset = extract_speaker_text_from_json_in_folder(test_path)

        if not dataset:
            print("오류: 데이터셋을 찾을 수 없습니다.")
            # 실제 배포 시에는 여기서 에러 처리를 해야 하지만, 일단 진행

        dataset_embeddings = model.encode(dataset, convert_to_tensor=True)
        torch.save(dataset_embeddings, EMBEDDING_FILE)
        with open(TEXT_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("준비 완료!")


# 4. 요청 데이터 형식 정의 (Node.js가 보낼 데이터)
class ChatRequest(BaseModel):
    user_input: str


# 5. 채팅 API 엔드포인트
@app.post("/chat")
async def chat(request: ChatRequest):
    global model, dataset_embeddings, dataset

    query = request.user_input

    # 임베딩 및 검색
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, dataset_embeddings, top_k=1)

    # 가장 유사한 결과 1개 가져오기
    top_hit = hits[0][0]
    matched_text = dataset[top_hit['corpus_id']]
    score = top_hit['score']

    # [수정됨] 답변만 추출하는 로직 추가
    if "답변:" in matched_text:
        # "답변:" 기준으로 자르고 뒷부분만 가져옴 + 공백 제거
        answer_only = matched_text.split("답변:", 1)[1].strip()
    else:
        # 혹시 형식이 다르면 전체 반환
        answer_only = matched_text

    return {
        "reply": answer_only,  # 질문 포함된 전체 텍스트 대신 답변만 반환
        "score": float(score)
    }

# 로컬 테스트용 (도커에서는 CMD로 실행하므로 주석 처리 가능)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)