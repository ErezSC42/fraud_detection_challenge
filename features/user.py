import os
import pickle
import numpy as np
import pandas as pd

from features.load import load_user_data
from constants import *
from commands import global_cmd_map_code, global_cmds
from features.segment import build_segment_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_VECTORIZER_PATH = "tfidf_vectorizer.pkl"


class User:
    def __init__(self,
                 user_id: str,
                 user_data_path: os.PathLike,
                 ground_truth_path: pd.Series):
        '''
        :param user_id:
        :param user_data_path:
        :param groud_truth_path:
        '''
        self._user_id = user_id
        self._user_data_path = user_data_path

        with open(TFIDF_VECTORIZER_PATH, "rb") as fp:
            self.tfidf_vectorizer: TfidfVectorizer = pickle.load(fp)

        #   load user data
        self._train_raw_data, self._test_raw_data = load_user_data(self._user_id, self._user_data_path)

        #   cmd sets
        self.cmd_set_train = set(self._train_raw_data["cmd"].unique())
        self.cmd_set_test = set(self._test_raw_data["cmd"].unique())
        self.cmd_set = self.cmd_set_train.union(self.cmd_set_test)
        self.cmd_set_not_in_train = self.cmd_set_test.difference(self.cmd_set_train)

        #   get anomaly ground truth

        self.user_anomaly_gt = ground_truth_path

        # cmd map features
        self._train_raw_data["cmd_code"] = self._train_raw_data["cmd"].map(global_cmd_map_code).astype(int)
        self._test_raw_data["cmd_code"] = self._test_raw_data["cmd"].map(global_cmd_map_code).astype(int)
        self.scaler = MinMaxScaler()

    @property
    def segment_df_train(self):
        return self._segment_df_train

    @property
    def segment_df_test(self):
        return self._segment_df_test

    @segment_df_train.setter
    def segment_df_train(self, other):
        self._segment_df_train = other

    @segment_df_test.setter
    def segment_df_test(self, other):
        self._segment_df_test = other

    def build_segment_features(self):
        train_segment_features_list = []
        test_segment_features_list = []

        #   train data
        for i in range(TRAIN_SEGMENT_COUNT):
            train_segment_features_list.append(build_segment_features(
                self._train_raw_data, self.cmd_set_not_in_train, i))

        #   test data
        for i in range(TEST_SEGMENT_COUNT):
            test_segment_features_list.append(build_segment_features(
                self._test_raw_data, self.cmd_set_not_in_train, i))

        #   build segments dataframes
        self._segment_df_train = pd.DataFrame.from_records(train_segment_features_list)
        self._segment_df_test = pd.DataFrame.from_records(test_segment_features_list)

        # convert non-numeric columns to categorical
        for categorical_feature in ["cmd_most_used", "first_cmd", "last_cmd"]:
            self._segment_df_train[categorical_feature] = pd.Categorical(self._segment_df_train[categorical_feature],
                                                                         categories=global_cmds)
            self._segment_df_test[categorical_feature] = pd.Categorical(self._segment_df_test[categorical_feature],
                                                                        categories=global_cmds)

        #   fit scaler
        self.normalize_features()
        # self.get_tfidf_features()


    def normalize_features(self):
        numeric_columns = self._segment_df_train.select_dtypes(include=[np.number]).columns
        self.scaler.fit(self._segment_df_train[numeric_columns])

        self.normalized_features_train = self._segment_df_train.copy(deep=True)
        self.normalized_features_test = self._segment_df_test.copy(deep=True)

        self.normalized_features_train[numeric_columns] = self.scaler.transform(self._segment_df_train[numeric_columns])
        self.normalized_features_test[numeric_columns] = self.scaler.transform(self._segment_df_test[numeric_columns])

    def get_tfidf_features(self):
        train_corpus = self._train_raw_data.groupby(["user", "segment_id"])["cmd"].transform(
            lambda x: ' '.join(x)).drop_duplicates().reset_index(drop=True).tolist()
        test_corpus = self._test_raw_data.groupby(["user", "segment_id"])["cmd"].transform(
            lambda x: ' '.join(x)).drop_duplicates().reset_index(drop=True).tolist()

        tfidf_features_train = pd.DataFrame(self.tfidf_vectorizer.transform(train_corpus).toarray())
        tfidf_features_test = pd.DataFrame(self.tfidf_vectorizer.transform(test_corpus).toarray())

        columns = [f"tfidf_{s}" for s in self.tfidf_vectorizer.get_feature_names_out().tolist()]
        tfidf_features_train.columns = columns
        tfidf_features_test.columns = columns
        # add tfidf features to segments df
        self.normalized_features_train = pd.concat([self.normalized_features_train, tfidf_features_train], axis=1)
        self.normalized_features_test = pd.concat([self.normalized_features_test, tfidf_features_test], axis=1)




    def get_dummies_features(self):
        pass


if __name__ == '__main__':
    user_id = "User0"
    user_data_path = "data/User0"
    gt_path = "challengeToFill.csv"

    user = User(user_id, user_data_path, gt_path)
