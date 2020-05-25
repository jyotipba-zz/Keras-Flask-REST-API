FROM python:3.7.3-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY . /app
CMD ["python", "/app/main.py"]
