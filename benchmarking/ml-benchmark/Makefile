# Run from benchmarking/ml-benchmark/ (Compose context = repo root).

COMPOSE_FILE := $(dir $(lastword $(MAKEFILE_LIST)))docker-compose.yml
COMPOSE := docker compose -f "$(COMPOSE_FILE)"

.PHONY: help build-ml build-ml-sgx run-quick run-standard run-with-client shell-ml shell-ml-sgx build-sentra-sgx

help:
	@echo "benchmarking/ml-benchmark — see README.md"
	@echo "  make build-ml          - sentra-ml-benchmark:latest"
	@echo "  make build-ml-sgx          - sentra-ml-benchmark-sgx:latest"
	@echo "  make run-quick|run-standard|run-with-client"
	@echo "  make shell-ml"
	@echo "  make shell-ml-sgx"
	@echo "  make build-sentra-sgx  - ../sentra-node/ Occlum image (long build)"

build-ml:
	$(COMPOSE) build ml-benchmark

build-ml-sgx:
	$(COMPOSE) build ml-benchmark-sgx

run-quick:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/quick.yaml

run-standard:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/standard.yaml

run-with-client:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/with-client.yaml

shell-ml:
	$(COMPOSE) run --rm ml-benchmark bash

shell-ml-sgx:
	$(COMPOSE) run --rm ml-benchmark-sgx bash

build-sentra-sgx:
	cd "$(dir $(lastword $(MAKEFILE_LIST)))../sentra-node" && sh ./buildSentraNodeDockerImage.sh
