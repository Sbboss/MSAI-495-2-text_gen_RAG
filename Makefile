PYTHON ?= python3

.PHONY: install data-download data-clean clean-hf-cache

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
