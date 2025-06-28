#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = e_commerce_customer_ai_platform
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies (after environment is created)
.PHONY: requirements
requirements:
	@echo ">>> Updating environment dependencies..."
	@conda env update --name $(PROJECT_NAME) --file environment.yml --prune



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to auto-fix)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Set up Python interpreter environment (run once)
.PHONY: create_environment
create_environment:
	@echo ">>> Creating new conda environment..."
	@conda env create --name $(PROJECT_NAME) -f environment.yml
	@echo ">>> Environment created. Activate with:\nconda activate $(PROJECT_NAME)"


## Download raw Olist data via Prefect (with dependency check)
.PHONY: ingest
ingest: requirements
	$(PYTHON_INTERPRETER) -c "from flows.ingest_olist import ingest_flow; ingest_flow()"

## Download raw Olist data via Prefect (fast, skip dependency check)
.PHONY: ingest-fast
ingest-fast:
	$(PYTHON_INTERPRETER) -c "from flows.ingest_olist import ingest_flow; ingest_flow()"

## clean the Olist data
clean-data:
	$(PYTHON_INTERPRETER) -m flows.clean_olist

## Load cleaned parquet to Postgres
.PHONY: db-load
db-load:
	$(PYTHON_INTERPRETER) -m flows.load_cleaned_to_db

## Build engineered features
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m flows.feature_builders

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Build processed dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) customer_ai/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Show this help message
.PHONY: help
help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)