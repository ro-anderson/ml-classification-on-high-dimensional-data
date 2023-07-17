from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from data_extractor import DataExtractor

class BaseDataVisualizer(ABC):
    def __init__(self, data):
        self.data = data
        self.results = None

    def prepare_data(self):
        embeddings = []
        labels = []
        for syndrome_id, subjects in self.data.items():
            for subject_id, images in subjects.items():
                for image_id, encoding in images.items():
                    embeddings.append(encoding)
                    labels.append(syndrome_id)
        return np.array(embeddings), labels

    @abstractmethod
    def perform_transformation(self, embeddings):
        pass

    @abstractmethod
    def visualize_data(self, labels):
        pass

    def run(self):
        embeddings, labels = self.prepare_data()
        self.perform_transformation(embeddings)
        self.visualize_data(labels)

class TSNEDataVisualizer(BaseDataVisualizer):
    def perform_transformation(self, embeddings):
        tsne = TSNE(n_components=2, random_state=42)
        self.results = tsne.fit_transform(embeddings)

    def visualize_data(self, labels):
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(set(labels)):
            indices = [idx for idx, l in enumerate(labels) if l == label]
            subset = self.results[indices]
            plt.scatter(subset[:, 0], subset[:, 1], label=label)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    de = DataExtractor()
    dv = TSNEDataVisualizer(de.data)
    dv.run()
