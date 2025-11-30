from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import os
import json
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI  # [추가] Ollama 연결용

# --- 기존 config 및 read_data 임포트 ---
from config import config
from read_data import extract_speaker_text_from_json_in_folder

# ==========================================
# [설정] Local LLM (Ollama) 연결 설정
# ==========================================
# 1. 사용할 모델 이름 (터미널에서 'ollama pull exaone3.5' 미리 실행 필요)
LOCAL_MODEL_NAME = "exaone3.5"

# 2. Ollama 주소 설정
# Docker에서 실행 시 -e OLLAMA_URL="..." 옵션으로 주입된 값을 사용
# 값이 없으면 로컬 기본값(localhost) 사용
default_url = "http://localhost:11434/v1"
OLLAMA_URL = os.getenv("OLLAMA_URL", default_url)

print(f"🔗 AI 연결 주소: {OLLAMA_URL}")

# Ollama 클라이언트 초기화
client = OpenAI(
    base_url=OLLAMA_URL,
    api_key="ollama"
)

app = FastAPI()

# 전역 변수 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
dataset_embeddings = None
dataset = []

EMBEDDING_FILE = config.EMBEDDING_FILE
TEXT_DATA_FILE = config.TEXT_DATA_FILE


@app.on_event("startup")
async def startup_event():
    global model, dataset_embeddings, dataset
    print(f"🚀 서버 시작! 장치: {device.upper()}")

    # 모델 로드
    model = SentenceTransformer('jhgan/ko-sbert-nli', device=device)

    # 데이터 로드 또는 생성
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(TEXT_DATA_FILE):
        print("--- 저장된 데이터 로드 중 ---")
        dataset_embeddings = torch.load(EMBEDDING_FILE, map_location=device)
        with open(TEXT_DATA_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        print("--- 데이터 생성 중 ---")
        test_path = os.path.join("dataset", "Training")
        dataset = extract_speaker_text_from_json_in_folder(test_path)

        if not dataset:
            print("❌ 오류: 데이터셋을 찾을 수 없습니다.")

        dataset_embeddings = model.encode(dataset, convert_to_tensor=True)
        torch.save(dataset_embeddings, EMBEDDING_FILE)
        with open(TEXT_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("✅ 준비 완료!")


class ChatRequest(BaseModel):
    user_input: str


@app.post("/chat")
async def chat(request: ChatRequest):
    global model, dataset_embeddings, dataset

    query = request.user_input

    # 1. [검색] SBERT로 가장 유사한 답변 찾기 (Retrieval)
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, dataset_embeddings, top_k=1)

    top_hit = hits[0][0]
    matched_text = dataset[top_hit['corpus_id']]
    score = top_hit['score']

    # 답변 부분만 추출 (Context로 사용)
    if "답변:" in matched_text:
        reference_answer = matched_text.split("답변:", 1)[1].strip()
    else:
        reference_answer = matched_text

    # 2. [생성] Ollama에게 답변 요약 요청 (Generation)
    print(f"🤖 {LOCAL_MODEL_NAME}에게 생성 요청 중...")

    try:
        completion = client.chat.completions.create(
            model=LOCAL_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 친절하고 전문적인 수의사입니다. "
                        "제공된 [참고 정보] 내용을 바탕으로 사용자의 질문에 답변하세요. "
                        "전문 용어는 쉽게 풀어서 설명하고, 3~4문장으로 핵심만 요약해서 따뜻하게 말해주세요."
                        "없는 내용은 지어내지 마세요."
                    )
                },
                {
                    "role": "user",
                    "content": f"사용자 질문: {query}\n\n[참고 정보]: {reference_answer}"
                }
            ],
            temperature=0.7
        )
        final_answer = completion.choices[0].message.content
        is_generated = True

    except Exception as e:
        print(f"❌ Ollama 연결 실패: {e}")
        # 실패 시 원본 답변 반환
        final_answer = reference_answer
        is_generated = False

    return {
        "reply": final_answer,
        "score": float(score),
        "is_generated": is_generated  # 생성 여부를 클라이언트가 알 수 있게 추가
    }

# 로컬 테스트용
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)