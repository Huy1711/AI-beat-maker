FROM python:3.10

WORKDIR /app

COPY ./requirements.txt ./

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

EXPOSE 8080

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]
