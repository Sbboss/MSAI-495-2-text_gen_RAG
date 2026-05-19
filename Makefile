PYTHON ?= python3

.PHONY: install data-download data-prepare index-build

install:
	$(PYTHON) -m pip install -r requirements.txt

data-download:
	$(PYTHON) scripts/download_datasets.py

data-prepare:
	$(PYTHON) scripts/prepare_retrieval_corpus.py

index-build:
	$(PYTHON) scripts/build_faiss_index.py
