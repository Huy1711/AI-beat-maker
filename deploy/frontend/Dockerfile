FROM python:3.10

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

EXPOSE 8081

CMD [ "nohup", "streamlit", "run", "main.py", "--server.maxUploadSize=2" ]
