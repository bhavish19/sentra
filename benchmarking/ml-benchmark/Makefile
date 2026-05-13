# Run from benchmarking/ml-benchmark/ (Compose context = repo root).

COMPOSE_FILE := $(dir $(lastword $(MAKEFILE_LIST)))docker-compose.yml
COMPOSE := docker compose -f "$(COMPOSE_FILE)"

.PHONY: help build-ml run-quick run-standard run-with-client shell-ml build-sentra-sgx

help:
	@echo "benchmarking/ml-benchmark — see README.md"
	@echo "  make build-ml          - sentra-ml-benchmark:latest"
	@echo "  make run-quick|run-standard|run-with-client"
	@echo "  make shell-ml"
	@echo "  make build-sentra-sgx  - ../sentra-node/ Occlum image (long build)"

build-ml:
	$(COMPOSE) build ml-benchmark

run-quick:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/quick.yaml

run-standard:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/standard.yaml

run-with-client:
	$(COMPOSE) run --rm ml-benchmark --config /workspace/benchmark-configs/with-client.yaml

shell-ml:
	$(COMPOSE) run --rm ml-benchmark bash

build-sentra-sgx:
	cd "$(dir $(lastword $(MAKEFILE_LIST)))../sentra-node" && sh ./buildSentraNodeDockerImage.sh
