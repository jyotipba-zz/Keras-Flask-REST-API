FROM python:3.7.3-slim
MAINTAINER Jyoti Bartaula <jyoti.bartaula@gmail.com>

WORKDIR usr/src/app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
copy . .
RUN chmod +x gunicorn_start.sh
EXPOSE 8003
ENTRYPOINT ["./gunicorn_start.sh"]
