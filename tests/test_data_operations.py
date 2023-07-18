import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from data_extractor import DataExtractor
from data_visualizer import TSNEDataVisualizer

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
