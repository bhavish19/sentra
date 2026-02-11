FROM python:3.14.3-slim-trixie AS sentra-backend

RUN mkdir /sentra-backend
COPY ./backend /sentra-backend
COPY ./demonstrator/ci/docker/scripts/entrypoint.sh /
RUN pip install -r requirements

ENTRYPOINT ["/entrypoint.sh"]
