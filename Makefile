PYTHON ?= python3

.PHONY: install data-download data-clean clean-hf-cache preprocess-so

install:
	$(PYTHON) -m pip install -r requirements.txt

data-download:
	$(PYTHON) scripts/download_datasets.py --max-so-shards 20

data-download-wiki:
	$(PYTHON) scripts/download_datasets.py --skip-so --with-wiki

data-clean:
	rm -rf data

clean-hf-cache:
	$(PYTHON) scripts/download_datasets.py --clear-cache-only

preprocess-so:
	$(PYTHON) scripts/preprocess_stackoverflow_stream.py --row-batch-size 64 --cleanup-parts

check-data:
	$(PYTHON) scripts/check_preprocess_layout.py

preprocess-so-fast:
	$(PYTHON) scripts/preprocess_stackoverflow_fast.py --memory-limit 80GB --workers 32 --encode-batch-size 4096 --cleanup-parts
