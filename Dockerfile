FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["main.py"]