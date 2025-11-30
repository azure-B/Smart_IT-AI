from config import config
from sentence_transformers import SentenceTransformer, util
from read_data import extract_speaker_text_from_json_in_folder
import torch
import os
import json
from openai import OpenAI  # [추가] Ollama 연결용 라이브러리

# 1. 사용할 모델 이름 (터미널에서 'ollama pull exaone3.5' 미리 실행 필요)
LOCAL_MODEL_NAME = "exaone3.5"

# 2. Ollama 주소 설정
default_url = "http://localhost:11434/v1"
OLLAMA_URL = os.getenv("OLLAMA_URL", default_url)

print(f"🔗 AI 연결 주소: {OLLAMA_URL}")

# Ollama 클라이언트 초기화
client = OpenAI(
    base_url=OLLAMA_URL,
    api_key="ollama"  # Ollama는 키가 필요 없지만 형식상 입력
)

# ==========================================
# 1. 장치 자동 설정
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("-" * 30)
if device == "cuda":
    print(f"CUDA 사용 중 ({torch.cuda.get_device_name(0)})")
else:
    print("CPU 사용 중")
print("-" * 30)

EMBEDDING_FILE = config.EMBEDDING_FILE
TEXT_DATA_FILE = config.TEXT_DATA_FILE

# 2. 모델 로드 (SBERT: 검색 담당)
model = SentenceTransformer('jhgan/ko-sbert-nli', device=device)

if os.path.exists(EMBEDDING_FILE) and os.path.exists(TEXT_DATA_FILE):
    print("--- 저장된 데이터를 로드합니다 ---")
    dataset_embeddings = torch.load(EMBEDDING_FILE, map_location=device)
    with open(TEXT_DATA_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

else:
    print("--- 데이터셋 생성 및 임베딩 시작 ---")
    test_path = os.path.join("../Dataset", "Training")
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
        user_speech = input(" 💬 입력 (종료하려면 '종료'): ")
    except EOFError:
        break

    if user_speech.strip():
        if "종료" in user_speech.replace(" ", ""):
            print(" 프로그램 종료 ")
            break

        # 1. [검색] 사용자의 질문과 가장 유사한 답변 찾기 (Retrieval)
        user_speech_embedding = model.encode(user_speech, convert_to_tensor=True)
        hits = util.semantic_search(user_speech_embedding, dataset_embeddings, top_k=1)

        # 가장 유사한 1개만 가져옵니다.
        top_hit = hits[0][0]
        matched_text = dataset[top_hit['corpus_id']]
        similarity_score = top_hit['score']

        # 답변 부분만 추출 (Context로 사용)
        if "답변:" in matched_text:
            reference_answer = matched_text.split("답변:", 1)[1].strip()
        else:
            reference_answer = matched_text

        print(f"\[참고 자료 검색 완료] (유사도: {similarity_score:.4f})")
        # 디버깅용으로 원본이 보고 싶으면 아래 주석 해제
        # print(f"참고 내용: {reference_answer[:100]}...")

        # 2. [생성] Ollama에게 요약 및 답변 생성 요청 (Generation)
        print(f"{LOCAL_MODEL_NAME} 수의사가 답변을 생성 중입니다...")

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
                        "content": f"사용자 질문: {user_speech}\n\n[참고 정보]: {reference_answer}"
                    }
                ],
                temperature=0.7  # 창의성 조절
            )
            final_answer = completion.choices[0].message.content

            print("\n[수의사 답변]:")
            print(final_answer)

        except Exception as e:
            print(f"\n Ollama 연결 실패: {e}")
            print("\n[원본 답변]:")
            print(reference_answer)

    else:
        print("내용을 입력해주세요.")

    print("=" * 30)