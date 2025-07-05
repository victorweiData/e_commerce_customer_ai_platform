#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME        = e_commerce_customer_ai_platform
PYTHON_VERSION      = 3.11
PYTHON_INTERPRETER  = python
RUN_FLOW            = $(PYTHON_INTERPRETER) -c   # helper for “python -c …”

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install / update Python dependencies
.PHONY: requirements
requirements:
	@echo ">>> Updating environment dependencies..."
	@conda env update --name $(PROJECT_NAME) --file environment.yml --prune


## Delete *.py[co] + __pycache__
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff
.PHONY: lint
lint:
	ruff format --check
	ruff check


## Auto-format with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## One-off: create the Conda environment
.PHONY: create_environment
create_environment:
	@echo ">>> Creating new conda environment…"
	@conda env create --name $(PROJECT_NAME) -f environment.yml
	@echo "Activate it with:\nconda activate $(PROJECT_NAME)"


# ----------------------------------------------------------------------------- #
# Data ingest / cleaning flows                                                  #
# ----------------------------------------------------------------------------- #

## Download the raw Olist dataset
.PHONY: ingest
ingest: requirements
	$(RUN_FLOW) "from flows.ingest_olist import ingest_flow; ingest_flow()"

## Faster ingest (skip env check)
.PHONY: ingest-fast
ingest-fast:
	$(RUN_FLOW) "from flows.ingest_olist import ingest_flow; ingest_flow()"

## Load cleaned parquet into Postgres
.PHONY: db-load
db-load:
	$(PYTHON_INTERPRETER) -m flows.load_raw_to_db


# ----------------------------------------------------------------------------- #
# 🔥  CHURN-SPECIFIC TARGETS                                                    #
# ----------------------------------------------------------------------------- #

## Build churn feature tables (train / val / test)
.PHONY: churn-features
churn-features:
	$(PYTHON_INTERPRETER) flows/build_customer_churn_master.py

## Train / evaluate churn model stand-alone (no Prefect)
.PHONY: churn-model
churn-model:
	$(PYTHON_INTERPRETER) scripts/build_customer_churn.py

## Train churn model **inside Prefect** (tracked in UI)
.PHONY: churn-model-flow
churn-model-flow:
	$(PYTHON_INTERPRETER) flows/build_customer_churn.py

## Full churn pipeline: features ➔ model (flow)
.PHONY: churn-all
churn-all: churn-features churn-model-flow


# ----------------------------------------------------------------------------- #
# 🔥  SEGMENTATION-SPECIFIC TARGETS                                             #
# ----------------------------------------------------------------------------- #

## Build segmentation feature tables
.PHONY: segment-features
segment-features:
	$(PYTHON_INTERPRETER) flows/build_customer_segments_master.py

## Train / evaluate segmentation model stand-alone
.PHONY: segment-model
segment-model:
	$(PYTHON_INTERPRETER) scripts/build_customer_segments.py

## Train segmentation model **inside Prefect**
.PHONY: segment-model-flow
segment-model-flow:
	$(PYTHON_INTERPRETER) flows/build_customer_segments.py

## Full segmentation pipeline: features ➔ model (flow)
.PHONY: segment-all
segment-all: segment-features segment-model-flow

## Start React dev server
.PHONY: dashboard-dev
dashboard-dev:
	cd dashboard && npm run dev


#################################################################################
# Self-documenting help                                                         #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
print("Available rules:\n")
for line in sys.stdin:
    if line.startswith("##"):
        desc = line[2:].strip()
        target = next(sys.stdin).split(":")[0].strip()
        print(f"{target:25} {desc}")
endef
export PRINT_HELP_PYSCRIPT

## Show this help message
.PHONY: help
help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)