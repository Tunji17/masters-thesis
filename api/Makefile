# Medical Graph Extraction API - Makefile
# This makefile manages the local development environment

# Variables
PYTHON := python3
PIP := pip3
VENV_PATH := .venv
NEO4J_COMPOSE := docker-compose.yml
SRC_DIR := ../src

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help setup setup-api setup-src run-neo4j run-api run-all stop clean test lint

help: ## Show this help message
	@echo "$(BLUE)Medical Graph Extraction API - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  setup         - Set up both API and notebook environments"
	@echo "  setup-api     - Set up API virtual environment and dependencies"
	@echo "  setup-src     - Set up source/notebook environment"
	@echo ""
	@echo "$(GREEN)Running:$(NC)"
	@echo "  run-neo4j     - Start Neo4j database container"
	@echo "  run-api       - Run API server locally (requires Neo4j)"
	@echo "  run-all       - Start Neo4j and API together"
	@echo "  stop          - Stop Neo4j container"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  test          - Run API tests"
	@echo "  lint          - Run code linting"
	@echo "  clean         - Clean up containers, volumes, and cache"

setup: setup-api setup-src ## Set up both API and notebook environments
	@echo "$(GREEN)✓ Setup complete for both API and notebook environments$(NC)"

setup-api: ## Set up API virtual environment and dependencies
	@echo "$(BLUE)Setting up API environment...$(NC)"
	$(PYTHON) -m venv .venv
	source .venv/bin/activate && $(PIP) install --upgrade pip setuptools wheel
	source .venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "$(BLUE)Installing SciSpacy model...$(NC)"
	source .venv/bin/activate && $(PIP) install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
	@echo "$(GREEN)✓ API environment setup complete$(NC)"

setup-src: ## Set up source/notebook environment
	@echo "$(BLUE)Setting up notebook environment...$(NC)"
	cd $(SRC_DIR) && $(PYTHON) -m venv .venv
	cd $(SRC_DIR) && source .venv/bin/activate && $(PIP) install --upgrade pip setuptools wheel
	cd $(SRC_DIR) && source .venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Notebook environment setup complete$(NC)"

run-neo4j: ## Start Neo4j database container
	@echo "$(BLUE)Starting Neo4j database...$(NC)"
	docker-compose up -d neo4j
	@echo "$(GREEN)✓ Neo4j started. Access Neo4j Browser at http://localhost:7474$(NC)"
	@echo "$(YELLOW)Default credentials: neo4j/password123$(NC)"

run-api: ## Run API server locally
	@echo "$(BLUE)Starting API server...$(NC)"
	@if ! docker-compose ps neo4j | grep -q "Up"; then \
		echo "$(YELLOW)Neo4j not running. Starting Neo4j first...$(NC)"; \
		make run-neo4j; \
		sleep 10; \
	fi
	@echo "$(BLUE)Waiting for Neo4j to be ready...$(NC)"
	@timeout=60; while [ $$timeout -gt 0 ] && ! docker-compose exec neo4j cypher-shell -u neo4j -p password123 "RETURN 1" >/dev/null 2>&1; do \
		sleep 2; \
		timeout=$$((timeout-2)); \
	done
	source .venv/bin/activate && \
		NEO4J_URI=bolt://localhost:7687 \
		NEO4J_USER=neo4j \
		NEO4J_PASSWORD=password123 \
		GEMMA_MODEL=google/gemma-3-4b-it \
		NER_MODEL=mlx-ner-medical \
		SCISPACY_MODEL=en_core_sci_lg \
		API_PORT=8000 \
		LOG_LEVEL=INFO \
		CACHE_SIZE=10000 \
		$(PYTHON) main.py

run-all: run-neo4j ## Start Neo4j and API together
	@echo "$(BLUE)Starting complete system...$(NC)"
	@sleep 5
	@make run-api

stop: ## Stop Neo4j container
	@echo "$(BLUE)Stopping Neo4j...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Neo4j stopped$(NC)"

test: ## Run API tests
	@echo "$(BLUE)Running tests...$(NC)"
	source .venv/bin/activate && $(PYTHON) -m pytest tests/ -v

lint: ## Run code linting
	@echo "$(BLUE)Running linting...$(NC)"
	source .venv/bin/activate && flake8 . --max-line-length=100 --exclude=.venv
	cd $(SRC_DIR) && source .venv/bin/activate && flake8 . --max-line-length=100 --exclude=.venv

clean: ## Clean up containers, volumes, and cache
	@echo "$(BLUE)Cleaning up...$(NC)"
	docker-compose down -v --remove-orphans
	docker system prune -f --volumes
	@echo "$(YELLOW)Removing Python cache files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find $(SRC_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(SRC_DIR) -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Development shortcuts
dev-api: run-neo4j ## Quick start for API development
	@sleep 5
	@make run-api

dev-notebook: ## Start Jupyter for notebook development
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	cd $(SRC_DIR) && source .venv/bin/activate && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

status: ## Show status of services
	@echo "$(BLUE)Service Status:$(NC)"
	@echo "$(YELLOW)Neo4j Container:$(NC)"
	@docker-compose ps neo4j || echo "Not running"
	@echo ""
	@echo "$(YELLOW)Neo4j Connection Test:$(NC)"
	@docker-compose exec neo4j cypher-shell -u neo4j -p password123 "RETURN 'Connected' as status" 2>/dev/null || echo "Cannot connect"