FROM conda/miniconda3

WORKDIR /app

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt
RUN python3 -c "import multipart; print(multipart.__version__)"

COPY . .

ENTRYPOINT ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

VOLUME /app/data

EXPOSE 8080