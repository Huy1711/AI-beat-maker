FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

EXPOSE 8081

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

CMD ["streamlit", "run", "Music_Generate.py", "--server.port", "8081"]
