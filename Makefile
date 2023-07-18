.PHONY: clean run_classifier test

clean:
	find ./data -type f -name "*.png" -delete
	find ./data -type f -name "*.csv" -delete
	find ./data -type f -name "*.txt" -delete
	find . -name "__pycache__" -type d -exec rm -r {} \+

run_classifier:
	python data_visualizer.py &&\
	python model_evaluation.py


test:
	pytest -v tests/test_data_operations.py
