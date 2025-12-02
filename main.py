from config import config
from sentence_transformers import SentenceTransformer, util
from read_data import extract_speaker_text_from_json_in_folder
import torch
import os
import json
from openai import OpenAI
import sys
import io

# [필수] 출력(Print)은 UTF-8로 강제 고정 (이모지 및 한글 출력용)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ==========================================
# [핵심 수정] 한글 입력 깨짐 방지 함수
# ==========================================
def safe_input(prompt):
    """
    윈도우 도커 환경에서 input() 사용 시 발생하는 UnicodeDecodeError를 방지합니다.
    데이터를 바이트(Raw Byte) 단위로 받아서 UTF-8 또는 CP949로 번역을 시도합니다.
    """
    print(prompt, end='', flush=True)
    try:
        # 1. 표준 입력 버퍼에서 날것의 데이터 읽기
        line = sys.stdin.buffer.readline()
        if not line: return ""  # EOF 처리

        # 2. UTF-8로 먼저 디코딩 시도 (대부분의 리눅스/도커 환경)
        try:
            return line.decode('utf-8').strip()
        except UnicodeDecodeError:
            # 3. 실패 시 윈도우 기본 인코딩(CP949)으로 디코딩 시도
            return line.decode('cp949').strip()
    except Exception:
        return ""


# ==========================================
# 설정 및 초기화
# ==========================================

# 1. 사용할 모델 이름
LOCAL_MODEL_NAME = "exaone3.5"

# 2. Ollama 주소 설정
default_url = "http://localhost:11434/v1"
OLLAMA_URL = os.getenv("OLLAMA_URL", default_url)

print(f"🔗 AI 연결 주소: {OLLAMA_URL}")

# Ollama 클라이언트 초기화
client = OpenAI(
    base_url=OLLAMA_URL,
    api_key="ollama"
)

# 3. 장치 자동 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

print("-" * 30)
if device == "cuda":
    print(f"CUDA 사용 중 ({torch.cuda.get_device_name(0)})")
else:
    print("CPU 사용 중")
print("-" * 30)

EMBEDDING_FILE = config.EMBEDDING_FILE
TEXT_DATA_FILE = config.TEXT_DATA_FILE

# 4. 모델 로드 (SBERT)
model = SentenceTransformer('jhgan/ko-sbert-nli', device=device)

if os.path.exists(EMBEDDING_FILE) and os.path.exists(TEXT_DATA_FILE):
    print("--- 저장된 데이터를 로드합니다 ---")
    dataset_embeddings = torch.load(EMBEDDING_FILE, map_location=device)
    with open(TEXT_DATA_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
else:
    print("--- 데이터셋 생성 및 임베딩 시작 ---")
    # 경로가 ./Dataset 인지 ../Dataset 인지 환경에 맞게 확인 필요 (현재 ./Dataset으로 수정됨)
    test_path = os.path.join("./Dataset", "Training")

    # 폴더가 없을 경우 예외처리
    if not os.path.exists(test_path):
        # 만약 도커에서 경로가 다르다면 ../Dataset으로 시도
        test_path = os.path.join("../Dataset", "Training")

    dataset = extract_speaker_text_from_json_in_folder(test_path)

    if not dataset:
        print(f"오류: 데이터셋을 찾을 수 없습니다. 경로: {test_path}")
        exit()

    dataset_embeddings = model.encode(dataset, convert_to_tensor=True)

    torch.save(dataset_embeddings, EMBEDDING_FILE)
    with open(TEXT_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"--- 데이터 준비 완료 (총 {len(dataset)}개) ---")
print("-" * 30)

# ==========================================
# 메인 루프
# ==========================================
while True:
    try:
        # [수정] input() 대신 safe_input() 사용
        user_speech = safe_input(" 💬 입력 (종료하려면 '종료'): ")
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        break

    if user_speech:  # 내용이 있을 때만 실행
        if "종료" in user_speech.replace(" ", ""):
            print(" 프로그램 종료 ")
            break

        # 1. [검색] (Retrieval)
        user_speech_embedding = model.encode(user_speech, convert_to_tensor=True)
        hits = util.semantic_search(user_speech_embedding, dataset_embeddings, top_k=1)

        top_hit = hits[0][0]
        matched_text = dataset[top_hit['corpus_id']]
        similarity_score = top_hit['score']

        if "답변:" in matched_text:
            reference_answer = matched_text.split("답변:", 1)[1].strip()
        else:
            reference_answer = matched_text

        print(f"\n[참고 자료 검색 완료] (유사도: {similarity_score:.4f})")

        # 2. [생성] (Generation)
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
                temperature=0.7
            )
            final_answer = completion.choices[0].message.content

            print("\n[수의사 답변]:")
            print(final_answer)

        except Exception as e:
            print(f"\n Ollama 연결 실패: {e}")
            print("\n[원본 답변 (Fallback)]:")
            print(reference_answer)

    else:
        # 엔터만 쳤을 때
        pass

    print("=" * 30)