import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data_extractor import DataExtractor

class MetricStrategy:
    def compute_metric(self, y_true, y_pred):
        raise NotImplementedError

class Precision(MetricStrategy):
    def compute_metric(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='weighted')

class Recall(MetricStrategy):
    def compute_metric(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='weighted')

class F1Score(MetricStrategy):
    def compute_metric(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

class Accuracy(MetricStrategy):
    def compute_metric(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

class DistanceCalculator:
    def __init__(self):
        self.de = DataExtractor()
        self.data = self.de.data
        self.metric_strategies = {'Precision': Precision(), 'Recall': Recall(), 'F1-Score': F1Score(), 'Accuracy': Accuracy()}
        self.performance_metrics = pd.DataFrame(columns=['Metric', 'Cosine Distance', 'Euclidean Distance'])


    def _prepare_data(self, data):
        embeddings = []
        labels = []
        for syndrome_id, syndrome_value in data.items():
            for subject_id, subject_value in syndrome_value.items():
                for image_id, image_embedding in subject_value.items():
                    # append the entire 320x1 embedding array as a single element of the list
                    embeddings.append(np.array(image_embedding))
                    # use the syndrome_id as the label
                    labels.append(syndrome_id)
        return embeddings, labels

    def perform_cross_validation(self):
        kfold = KFold(n_splits=10)
        data, labels = self._prepare_data(self.data)
        data = np.array(data)
        labels = np.array(labels)
        for train_index, test_index in kfold.split(data):
            train_data, test_data = data[train_index], data[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            self.calculate_distance(train_data, test_data)
            self.classify(train_data, test_data, train_labels, test_labels)

    def calculate_distance(self, train_data, test_data):
        cosine_distance = cosine_distances(train_data, test_data)
        euclidean_distance = euclidean_distances(train_data, test_data)
        print(f"Cosine Distance: {cosine_distance}")
        print(f"Euclidean Distance: {euclidean_distance}")

    def compute_performance_metrics(self, y_true, y_pred_cosine, y_pred_euclidean):
        for metric_name, strategy in self.metric_strategies.items():
            result_cosine = strategy.compute_metric(y_true, y_pred_cosine)
            result_euclidean = strategy.compute_metric(y_true, y_pred_euclidean)

            # Round the float values in d_
            d_ = {'Metric': metric_name, 'Cosine Distance': result_cosine, 'Euclidean Distance': result_euclidean}
            rounded_values = {k: round(v, 4) if isinstance(v, np.float64) else v for k, v in d_.items()}
            
            # Create a DataFrame from the rounded values
            new_row = pd.DataFrame([rounded_values])

            # Concatenate the DataFrames
            #self.performance_metrics = self.performance_metrics.append({'Metric': metric_name, 'Cosine Distance': result_cosine, 'Euclidean Distance': result_euclidean}, ignore_index=True)
            self.performance_metrics = pd.concat([self.performance_metrics, new_row], ignore_index=True)

    #def classify(self, train_data, test_data, train_labels, test_labels):
    #    knn_cosine = KNeighborsClassifier(metric='cosine')
    #    knn_euclidean = KNeighborsClassifier(metric='euclidean')
    #    knn_cosine.fit(train_data, train_labels)
    #    knn_euclidean.fit(train_data, train_labels)
    #    prediction_cosine = knn_cosine.predict(test_data)
    #    prediction_euclidean = knn_euclidean.predict(test_data)
    #    print(f"Predictions using Cosine distance: {prediction_cosine}")
    #    print(f"Predictions using Euclidean distance: {prediction_euclidean}")
    #    self.compute_performance_metrics(test_labels, prediction_cosine)
    #    self.compute_performance_metrics(test_labels, prediction_euclidean)
    #    self.compute_performance_metrics(test_labels, prediction_cosine, prediction_euclidean)
    #    print(self.performance_metrics)

    def classify(self, train_data, test_data, train_labels, test_labels):
        # Define classifiers
        knn_cosine = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        knn_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

        # Fit the models
        knn_cosine.fit(train_data, train_labels)
        knn_euclidean.fit(train_data, train_labels)

        # Predict the test set results
        prediction_cosine = knn_cosine.predict(test_data)
        prediction_euclidean = knn_euclidean.predict(test_data)

        # Compute performance metrics and add them to the DataFrame
        self.compute_performance_metrics(test_labels, prediction_cosine, prediction_euclidean)

        # Print the DataFrame
        print(self.performance_metrics)



if __name__ == '__main__':
    dc = DistanceCalculator()
    dc.perform_cross_validation()
