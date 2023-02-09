FROM tensorflow/tensorflow

RUN apt-get update
RUN apt-get install -y --no-install-recommends

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN python3 -c "import multipart; print(multipart.__version__)"

COPY . .

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

VOLUME /app/data

EXPOSE 8080