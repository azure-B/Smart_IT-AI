from config import config

from sentence_transformers import SentenceTransformer, util
from read_data import extract_speaker_text_from_json_in_folder

import torch
import os
import json

EMBEDDING_FILE = config.EMBEDDING_FILE
TEXT_DATA_FILE = config.TEXT_DATA_FILE

model = SentenceTransformer('jhgan/ko-sbert-nli')

if os.path.exists(EMBEDDING_FILE) and os.path.exists(TEXT_DATA_FILE):
    dataset_embeddings = torch.load(EMBEDDING_FILE)
    with open(TEXT_DATA_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

else:
    print("data set")
    test_path = os.path.join("dataset", "Training")
    dataset = extract_speaker_text_from_json_in_folder(test_path)
    dataset_embeddings = model.encode(dataset, convert_to_tensor=True)
    torch.save(dataset_embeddings, EMBEDDING_FILE)
    with open(TEXT_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

print("-" * 30)

while True:
    user_speech = input(" 입력 ")

    if user_speech:
        if "종료" in user_speech.replace(" ", ""):
            print(" 프로그램 종료 ")
            break
        # augmented_user_speeches = eda_model([user_speech])
        user_speech_embedding = model.encode(user_speech, convert_to_tensor=True)
        hits = util.semantic_search(user_speech_embedding, dataset_embeddings, top_k=3)
        print(user_speech)

        for hit in hits[0]:
            matched_sentence = dataset[hit['corpus_id']]
            similarity_score = hit['score']
            print(f"- \"{matched_sentence}\" (유사도: {similarity_score:.4f})")
    else:
        input("인식하지 못했습니다. 다시 시도해주세요.")

    print("-" * 30)