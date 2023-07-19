import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from data_extractor import DataExtractor
from data_visualizer import TSNEDataVisualizer
import collections
from sklearn.model_selection import StratifiedKFold
from data_extractor import DataExtractor

def test_data_extractor_loads_data():
    # Arrange
    de = DataExtractor()
    # Act
    data = de.data
    # Assert
    assert data is not None, "Data should not be None"
    assert len(data) > 0, "Data should not be empty"

def test_tsne_visualizer_prepare_data():
    # Arrange
    de = DataExtractor()
    dv = TSNEDataVisualizer(de.data)
    # Act
    embeddings, labels = dv.prepare_data()
    # Assert
    assert embeddings is not None, "Embeddings should not be None"
    assert labels is not None, "Labels should not be None"
    assert len(embeddings) > 0, "Embeddings should not be empty"
    assert len(labels) > 0, "Labels should not be empty"
    assert len(embeddings) == len(labels), "Number of embeddings and labels should be equal"

def test_stratified_kfold_distribution():
    de = DataExtractor()

    # Preparing embeddings and labels
    embeddings = []
    labels = []
    for syndrome_id, subjects in de.data.items():
        for subject_id, images in subjects.items():
            for image_id, encoding in images.items():
                embeddings.append(encoding)
                labels.append(syndrome_id)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    k = 5
    skf = StratifiedKFold(n_splits=k)

    for train_index, test_index in skf.split(embeddings, labels):
        train_labels = labels[train_index]
        test_labels = labels[test_index]

        train_counter = collections.Counter(train_labels)
        test_counter = collections.Counter(test_labels)

        for key in train_counter:
            assert abs(train_counter[key]/len(train_labels) - test_counter[key]/len(test_labels)) < 0.1, \
            "StratifiedKFold did not distribute classes evenly among folds."