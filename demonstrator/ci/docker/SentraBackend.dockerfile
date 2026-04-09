FROM python:3.13.12-slim AS sentra-backend
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade --yes && DEBIAN_FRONTEND=noninteractive apt-get install --yes curl pebble
RUN curl -fsSL https://deb.nodesource.com/setup_24.x -o /tmp/nodesource_setup.sh
RUN bash /tmp/nodesource_setup.sh
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes nodejs
RUN npm install --no-audit --no-fund -g npm@11.12.1
RUN pip install --upgrade pip

RUN mkdir /pebble
COPY ./demonstrator/ci/docker/config/pebble/ /pebble/

RUN mkdir -p /sentra/backend
RUN mkdir /sentra/frontend

COPY ./demonstrator/backend /sentra/backend
COPY ./demonstrator/ci/docker/scripts/entrypoint.sh /

RUN mkdir /sentra/ml_training
COPY ./sentra-node/python/ml_training /sentra/ml_training/
COPY ./sentra-node/python/pyproject.toml /sentra/

COPY ./sentra-node/docker/training_config.yaml /etc/

WORKDIR /sentra/backend
RUN python -m pip install -r requirements.txt
WORKDIR /sentra/
RUN python -m pip install .

COPY --exclude=dist --exclude=node_modules ./demonstrator/frontend /sentra/frontend/
WORKDIR /sentra/frontend

RUN rm package-lock.json
RUN npm install --no-audit --no-fund
RUN npm install --only=dev --no-audit --no-fund
RUN npm run build


ENTRYPOINT ["/entrypoint.sh"]
