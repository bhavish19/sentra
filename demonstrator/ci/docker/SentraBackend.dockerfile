FROM python:3.14.3-slim-trixie AS sentra-backend
RUN pip install --upgrade pip
RUN mkdir /sentra-backend

COPY ./demonstrator/backend /sentra-backend
COPY ./demonstrator/ci/docker/scripts/entrypoint.sh /

WORKDIR /sentra-backend
RUN pip install -r requirements.txt

ENTRYPOINT ["/entrypoint.sh"]
