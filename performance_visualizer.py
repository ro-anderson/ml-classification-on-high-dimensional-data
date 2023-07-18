from abc import ABC, abstractmethod
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import LabelEncoder

class BasePerformanceVisualizer(ABC):
    def __init__(self, y_true, y_proba):
        self.y_true = y_true
        self.y_proba = y_proba

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save(self, filename):
        pass


class RocAucVisualizer(BasePerformanceVisualizer):
    def __init__(self, y_true, y_proba, n_classes):
        super().__init__(y_true, y_proba)

        # Encode labels
        self.le = LabelEncoder()
        self.y_true_encoded = self.le.fit_transform(self.y_true)
        
        self.classes_with_positives = set(self.y_true_encoded)
        self.y_true_bin = label_binarize(self.y_true_encoded, classes=range(n_classes))
        self.fpr = dict()
        self.tpr = dict()
        self.auc_score = dict()
        
        print("Number of classes: ", n_classes)
        print("Classes with positive samples: ", self.classes_with_positives)

        for i in range(n_classes):
            if i in self.classes_with_positives:
                self.fpr[i], self.tpr[i], _ = roc_curve(self.y_true_bin[:, i], self.y_proba[:, i])
                self.auc_score[i] = auc(self.fpr[i], self.tpr[i])
                print(f"Class {i} - FPR: {self.fpr[i]}, TPR: {self.tpr[i]}, AUC: {self.auc_score[i]}")

    def plot(self):
        if not self.auc_score:
            print("No ROC curves generated. No classes with positive samples.")
            return
            
        for i in self.auc_score.keys():
            plt.figure()
            plt.plot(self.fpr[i], self.tpr[i], color='darkorange', lw=2, 
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, self.auc_score[i]))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic of class {0}'.format(i))
            plt.legend(loc="lower right")
            plt.show()

    def save(self, filename):
        if not self.auc_score:
            print("No ROC curves generated. No classes with positive samples.")
            return

        for i in self.auc_score.keys():
            plt.figure()
            plt.plot(self.fpr[i], self.tpr[i], color='darkorange', lw=2, 
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(i, self.auc_score[i]))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic of class {0}'.format(i))
            plt.legend(loc="lower right")
            plt.savefig(filename + "_class_{0}.png".format(i))
            plt.close()  # Close the current figure

        print("ROC curve images saved.")