from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import os
import json
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import sys
import io

# [필수] Docker 로그 출력을 위한 UTF-8 강제 설정 (한글 깨짐 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- config 및 read_data 임포트 ---
from config import config
from read_data import extract_speaker_text_from_json_in_folder

# ==========================================
# [설정] Local LLM (Ollama) 연결 설정
# ==========================================
LOCAL_MODEL_NAME = "exaone3.5"

# Docker 환경변수 OLLAMA_URL 사용 (없으면 로컬 기본값)
default_url = "http://localhost:11434/v1"
OLLAMA_URL = os.getenv("OLLAMA_URL", default_url)

print(f"🔗 AI 연결 주소: {OLLAMA_URL}")

client = OpenAI(
    base_url=OLLAMA_URL,
    api_key="ollama"
)

app = FastAPI()

# 전역 변수
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

        # [핵심 수정] 데이터셋 경로 자동 탐색 (./Dataset 또는 ../Dataset)
        folder_candidates = ["Dataset", "dataset"]
        base_paths = [".", ".."]
        found_path = None

        for base in base_paths:
            for folder in folder_candidates:
                candidate = os.path.join(base, folder, "Training")
                if os.path.exists(candidate):
                    found_path = candidate
                    break
            if found_path: break

        if not found_path:
            found_path = os.path.join("./Dataset", "Training")
            print(f"❌ 경고: 데이터셋 폴더를 찾지 못했습니다. 경로 확인 필요: {found_path}")

        dataset = extract_speaker_text_from_json_in_folder(found_path)

        if not dataset:
            print("❌ 오류: 로드된 데이터가 없습니다.")
            dataset = []
        else:
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

    query = request.user_input.strip()
    if not query:
        return {"reply": "내용을 입력해주세요.", "score": 0.0, "is_generated": False}

    if dataset_embeddings is None or len(dataset) == 0:
        return {"reply": "서버 초기화 중입니다. 잠시 후 다시 시도해주세요.", "score": 0.0, "is_generated": False}

    # 1. [검색] (Retrieval)
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, dataset_embeddings, top_k=1)

    top_hit = hits[0][0]
    matched_text = dataset[top_hit['corpus_id']]
    score = top_hit['score']

    if "답변:" in matched_text:
        reference_answer = matched_text.split("답변:", 1)[1].strip()
    else:
        reference_answer = matched_text

    # 2. [생성] (Generation)
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
        final_answer = reference_answer
        is_generated = False

    return {
        "reply": final_answer,
        "score": float(score),
        "is_generated": is_generated
    }


# [수정] Node.js(8008)와 겹치지 않게 5000번 포트로 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)