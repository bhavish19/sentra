# Convenience targets for non-SGX Docker benchmarks.
# Requires Docker Compose v2+. Paths are relative to this Makefile's directory.

COMPOSE_FILE := $(dir $(lastword $(MAKEFILE_LIST)))docker-compose.yml
COMPOSE := docker compose -f "$(COMPOSE_FILE)"

.PHONY: help build-nosgx run-quick run-standard run-with-client shell

help:
	@echo "SENTRA non-SGX benchmark (see README.md in this directory)"
	@echo "  make build-nosgx       - build sentra-benchmark:nosgx"
	@echo "  make run-quick         - nosgx-quick.yaml (default smoke)"
	@echo "  make run-standard      - nosgx-standard.yaml (longer, XLSX metrics)"
	@echo "  make run-with-client   - nosgx-with-client.yaml (client distributor)"
	@echo "  make shell             - interactive shell in the image"

build-nosgx:
	$(COMPOSE) build benchmark-nosgx

run-quick:
	$(COMPOSE) run --rm benchmark-nosgx --config /workspace/benchmark-configs/nosgx-quick.yaml

run-standard:
	$(COMPOSE) run --rm benchmark-nosgx --config /workspace/benchmark-configs/nosgx-standard.yaml

run-with-client:
	$(COMPOSE) run --rm benchmark-nosgx --config /workspace/benchmark-configs/nosgx-with-client.yaml

shell:
	$(COMPOSE) run --rm benchmark-nosgx bash
