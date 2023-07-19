.PHONY: clean run_classifier test all help

SHELL=/bin/bash                                                                                                                                                                                             
.DEFAULT_GOAL := help   

## Delete all .png, .csv, .txt and cache files
clean:
	find ./data -type f -name "*.png" -delete
	find ./data -type f -name "*.csv" -delete
	find ./data -type f -name "*.txt" -delete
	find . -name "__pycache__" -type d -exec rm -r {} \+


## Plot a 2D data visualizatiom, calculate distances, perform cross validation,handle data preparation and classifie the data.
run_classifier:
	python data_visualizer.py &&\
	python model_evaluation.py


## Run unit tests.
test:
	pytest -v tests/test_data_operations.py || (echo "Tests failed, not running classifier."; exit 1)

all: test run_classifier

## Shows this help text
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
