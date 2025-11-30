import os
import json


def extract_speaker_text_from_json_in_folder(folder_path):
    """
    폴더 내 JSON 파일을 순회하며 'qa' 데이터를 추출하여
    SBERT 임베딩을 위한 텍스트 리스트(List[str])를 반환합니다.

    형식: "질문: {내용} \n 답변: {내용}"
    """
    dataset = []

    if not os.path.isdir(folder_path):
        print(f"오류: 지정된 폴더 '{folder_path}'를 찾을 수 없습니다.")
        return []

    print(f"--- 폴더 '{folder_path}' 데이터 로드 중 ---")

    for item_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, item_name)

        if os.path.isfile(file_path) and file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 파일 내용이 리스트인 경우와 단일 딕셔너리인 경우 모두 처리
                    data_list = data if isinstance(data, list) else [data]

                    for entry in data_list:
                        # 'qa' 키가 있는 데이터만 처리
                        if isinstance(entry, dict) and "qa" in entry:
                            qa_content = entry["qa"]

                            # 질문(input)과 답변(output) 추출
                            user_input = qa_content.get("input", "").strip()
                            ai_output = qa_content.get("output", "").strip()

                            # 데이터가 비어있지 않은 경우에만 처리
                            if user_input and ai_output:
                                # 검색 시 질문과 답변의 문맥을 모두 고려하고,
                                # 결과 출력 시 내용을 한눈에 보기 위해 하나의 문자열로 합칩니다.
                                combined_text = (
                                    f"질문: {user_input}\n"
                                    f"답변: {ai_output}"
                                )
                                dataset.append(combined_text)

            except json.JSONDecodeError:
                print(f"경고: {file_path} 형식이 잘못되었습니다.")
            except Exception as e:
                print(f"경고: {file_path} 처리 중 오류: {e}")

    print(f"--- 데이터 로드 완료: 총 {len(dataset)}건 ---")
    return dataset