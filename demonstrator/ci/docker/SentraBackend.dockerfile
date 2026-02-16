FROM python:3.14.3-slim-trixie AS sentra-backend
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade --yes && DEBIAN_FRONTEND=noninteractive apt-get install --yes curl pebble
RUN curl -fsSL https://deb.nodesource.com/setup_24.x -o /tmp/nodesource_setup.sh
RUN bash /tmp/nodesource_setup.sh
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes nodejs
RUN npm install --no-audit --no-fund -g npm@11.4.2
RUN pip install --upgrade pip

RUN mkdir/pebble
COPY ./demonstrator/ci/docker/config/pebble/ /pebble/

RUN mkdir -p /sentra/backend
RUN mkdir /sentra/frontend

COPY ./demonstrator/backend /sentra/backend
COPY ./demonstrator/ci/docker/scripts/entrypoint.sh /

WORKDIR /sentra/backend
RUN pip install -r requirements.txt

COPY --exclude=dist --exclude=node_modules ./demonstrator/frontend /sentra/frontend/
WORKDIR /sentra/frontend

RUN rm package-lock.json
RUN npm install --no-audit --no-fund
RUN npm install --only=dev --no-audit --no-fund
RUN npm run build


ENTRYPOINT ["/entrypoint.sh"]
