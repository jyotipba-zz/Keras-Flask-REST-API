FROM python:3.7.3-slim
# set workdir
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY . /app
RUN chmod +x /app/gunicorn_start.sh
ENTRYPOINT ["/app/gunicorn_start.sh"]
