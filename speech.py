import whisper
import speech_recognition as sr
import numpy as np
import os
from gtts import gTTS
from playsound import playsound


# --- 1. 음성 인식을 담당하는 함수 (이전과 동일) ---
def transcribe_from_mic(model_name="base"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n주변 소음 측정 중... (3초간 조용히 해주세요)")
        r.adjust_for_ambient_noise(source, duration=3)
        print("이제 말씀해주세요! (종료하려면 '종료'라고 말하세요)")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("시간 초과: 아무런 음성도 감지되지 않았습니다.")
            return ""  # 빈 문자열 반환

    print("음성 감지 완료. 인식을 시작합니다...")
    try:
        raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        model = whisper.load_model(model_name)
        result = model.transcribe(audio_np, language="ko")
        return result["text"]
    except Exception as e:
        print(f"음성 인식 오류: {e}")
        return ""


# --- 2. 텍스트를 음성으로 변환하는 함수 ---
def speak(text):
    """주어진 텍스트를 음성으로 출력합니다."""
    try:

        tts = gTTS(text=text, lang='ko')
        tts.save("exam.mp3")
        playsound("exam.mp3")
        os.remove("exam.mp3")

    except Exception as e:
        print(f"음성 출력 오류: {e}")
