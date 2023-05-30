import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from eval.metrics import detection_metrics


class DetectionPipeline:
    def __init__(self):
        self.detector = LocalOutlierFactor(
            n_neighbors=11,
            novelty=True,
            leaf_size=2,
            algorithm="kd_tree",
            p=1,
            # contamination=0.5
        )

    def fit(self,
            df_train: pd.DataFrame):
        self.features_train_df = pd.get_dummies(df_train)
        self.detector.fit(self.features_train_df)

    def predict(self, df_test: pd.DataFrame):
        self.features_test_df = pd.get_dummies(df_test)
        predicted_anomalies = self.detector.predict(self.features_test_df)
        return predicted_anomalies

    def eval(self, predicted_, ground_truth: pd.Series):
        return detection_metrics(predicted_, ground_truth)
