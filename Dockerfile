FROM python:3.10-slim

WORKDIR /app

# 1. 라이브러리 설치 (AI + 서버 기능 모두 포함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 소스코드 전체 복사 (main.py, server.py 다 들어감)
COPY . .

# 3. 기본 실행 명령 (이건 docker-compose에서 덮어쓸 거라 비워둬도 되지만, 기본값으로 둡니다)
CMD ["python", "main.py"]