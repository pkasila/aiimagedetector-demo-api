FROM tensorflow/tensorflow

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["python", "-u", "/app/main.py"]

VOLUME /app/data