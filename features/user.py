import os

import pandas as pd

from features.load import load_user_data
from constants import *
from commands import global_cmd_map_code, global_cmds
from features.segment import build_segment_features


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

    @property
    def segment_df_train(self):
        return self._segment_df_train

    @property
    def segment_df_test(self):
        return self._segment_df_test

    def get_dummies_features(self):
        pass


if __name__ == '__main__':
    user_id = "User0"
    user_data_path = "data/User0"
    gt_path = "challengeToFill.csv"

    user = User(user_id, user_data_path, gt_path)
