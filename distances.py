import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from data_extractor import DataExtractor
from performance_visualizer import RocAucVisualizer
from datetime import date

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

class TopKAccuracy(MetricStrategy):
    def __init__(self, k):
        self.k = k

    def compute_metric(self, y_true, y_pred_proba):

        # Get the top k predictions by probability
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -self.k:]
        print(f"top_k_preds:\n{top_k_preds}\n")

        # We want to compare the indices of the top-k predictions
        # to the indices (classes) in y_true. So let's encode y_true 
        # into class indices as well.
        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true)
        print(f"y_true_encoded:\n{y_true_encoded}\n")

        # Check if y_true is in top k for each set of predictions
        mask = np.any(top_k_preds == y_true_encoded[:, None], axis=1)

        # Compute top k accuracy
        top_k_accuracy = np.mean(mask)
        return top_k_accuracy 

class AUC(MetricStrategy):
    def compute_metric(self, y_true, y_pred_proba):
        lb = LabelBinarizer()
        lb.fit(y_true)
        binarized_labels = lb.transform(y_true)
        auc_score = roc_auc_score(binarized_labels, y_pred_proba, multi_class='ovr')
        return auc_score

class DistanceCalculator:
    def __init__(self, model_name, validation_type, distance_metrics):
        self.model_name = model_name
        self.validation_type = validation_type
        self.distance_metrics = distance_metrics

        self.de = DataExtractor()
        self.data = self.de.data
        self.metric_strategies = {
            'Precision': Precision(), 
            'Recall': Recall(), 
            'F1-Score': F1Score(), 
            'Accuracy': Accuracy(), 
            'Top-3 Accuracy': TopKAccuracy(k=3), 
            'AUC': AUC()
        }

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
        skf = StratifiedKFold(n_splits=10)
        data, labels = self._prepare_data(self.data)
        data = np.array(data)
        labels = np.array(labels)
        for train_index, test_index in skf.split(data,labels):
            train_data, test_data = data[train_index], data[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            self.calculate_distance(train_data, test_data)
            self.classify(train_data, test_data, train_labels, test_labels)

    def calculate_distance(self, train_data, test_data):
        cosine_distance = cosine_distances(train_data, test_data)
        euclidean_distance = euclidean_distances(train_data, test_data)
        print(f"Cosine Distance: {cosine_distance}")
        print(f"Euclidean Distance: {euclidean_distance}")

    def compute_performance_metrics(self, y_true, y_pred_cosine, y_pred_euclidean, y_pred_cosine_proba, y_pred_euclidean_proba):
        for metric_name, strategy in self.metric_strategies.items():
            if metric_name in ['Precision', 'Recall', 'F1-Score', 'Accuracy']:
                result_cosine = strategy.compute_metric(y_true, y_pred_cosine)
                result_euclidean = strategy.compute_metric(y_true, y_pred_euclidean)
            else:
                result_cosine = strategy.compute_metric(y_true, y_pred_cosine_proba)
                result_euclidean = strategy.compute_metric(y_true, y_pred_euclidean_proba)

            # Round the float values in d_
            d_ = {'Metric': metric_name, 'Cosine Distance': result_cosine, 'Euclidean Distance': result_euclidean}
            rounded_values = {k: round(v, 4) if isinstance(v, np.float64) else v for k, v in d_.items()}

            # Create a DataFrame from the rounded values
            new_row = pd.DataFrame([rounded_values])

            # Concatenate the DataFrames
            self.performance_metrics = pd.concat([self.performance_metrics, new_row], ignore_index=True)

        n_classes = len(np.unique(y_true))

        # ROC AUC for Cosine Distance
        # print("y_true: ", y_true)
        # print("y_pred_cosine_proba: ", y_pred_cosine_proba)
        # print("y_pred_euclidean_proba: ", y_pred_euclidean_proba)
        roc_visualizer_cosine = RocAucVisualizer(y_true, y_pred_cosine_proba, n_classes)
        print("ROC AUC for Cosine Distance:")
        roc_visualizer_cosine.save("./data/roc_auc_cosine")

        # ROC AUC for Euclidean Distance
        roc_visualizer_euclidean = RocAucVisualizer(y_true, y_pred_euclidean_proba, n_classes)
        print("ROC AUC for Euclidean Distance:")
        roc_visualizer_euclidean.save("./data/roc_auc_euclidean")

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

        # Predict the test set probabilities
        prediction_cosine_proba = knn_cosine.predict_proba(test_data)
        prediction_euclidean_proba = knn_euclidean.predict_proba(test_data)

        # Compute performance metrics and add them to the DataFrame
        self.compute_performance_metrics(test_labels, prediction_cosine, prediction_euclidean, prediction_cosine_proba, prediction_euclidean_proba)

        # Print the DataFrame
        print(self.performance_metrics)

        # Export metrics as csv/txt
        for file_type in ['csv', 'txt']:

            self.performance_metrics.to_csv(f'./data/{self.generate_filename(file_type)}')
            self.performance_metrics.groupby('Metric').mean().round(4).to_csv(f'./data/Average_{self.generate_filename(file_type)}')

    def generate_filename(self, file_type=None):
        current_date = date.today().strftime('%Y%m%d')  # Get current date and format it as 'yyyymmdd'
        
        filename = f'{self.model_name}_Metrics_{self.validation_type}_{self.distance_metrics}_{current_date}.{file_type}'
        
        return filename 


if __name__ == '__main__':
    model_name = 'KNN'
    validation_type = 'StratifiedKFold'
    distance_metrics = 'CosineEuclidean'
    dc = DistanceCalculator(model_name, validation_type, distance_metrics)
    dc.perform_cross_validation()
