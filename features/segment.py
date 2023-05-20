import pandas as pd
from typing import Dict, Any, Set
from constants import SEGMENT_LEN
from commands import *


def set_feature_intersection_count(df: pd.DataFrame, column: str, cmd_set: Set[str]) -> int:
    return len(set(df[column]).intersection(cmd_set))


def build_segment_features(
        user_data_df: pd.DataFrame,
        user_cmd_set_not_in_train: Set[str],
        segment_id: int) -> Dict[str, Any]:
    '''
    build features per segment:
        1.  V - cmd that has been used the most
        2.  X - cmd that has been used the least
        3.  X - count of cmd that has been used the most
        4.  V - len of longest sequence of same cmds
        5.  X - has zip/compression cmds
        6.  X - has encryption cmds
        7.  X - has networking cmds
        8.  V - count distinct of cmds
        9.  V - first cmd
        10. V - last cmd
        11. V - commands that user never used in training count
    '''

    segment_features = {}

    segment_df = user_data_df[user_data_df["segment_id"] == segment_id]
    segment_cmd_value_counts = segment_df["cmd"].value_counts().to_dict()
    segment_df["cmd_count"] = segment_df["cmd"].map(segment_cmd_value_counts)

    # features
    segment_features["cmd_most_used"] = next(iter(segment_cmd_value_counts))
    segment_features["first_cmd"] = segment_df["cmd"].iloc[0]
    segment_features["last_cmd"] = segment_df["cmd"].iloc[SEGMENT_LEN - 1]
    segment_features["unique_cmds"] = len(segment_df["cmd"].unique())

    # longest subsequence of same commands
    s = segment_df["cmd_code"].diff().astype(bool)
    segment_features["longest_same_cmd_sequence"] = (~s).cumsum()[
        s].value_counts().max()  # TODO - there is a bug here, getting 100s

    segment_features["cmd_not_in_train_count"] = set_feature_intersection_count(segment_df, "cmd",
                                                                                user_cmd_set_not_in_train)
    segment_features["single_chars_cmd_count"] = set_feature_intersection_count(segment_df, "cmd",
                                                                                single_chars_cmds)
    segment_features["two_chars_cmds_count"] = set_feature_intersection_count(segment_df, "cmd", two_chars_cmds)
    segment_features["three_chars_cmds_count"] = set_feature_intersection_count(segment_df, "cmd", three_chars_cmds)
    segment_features["four_chars_cmds_count"] = set_feature_intersection_count(segment_df, "cmd", four_chars_cmds)
    segment_features["ends_with_dot_cmds_count"] = set_feature_intersection_count(segment_df, "cmd", ends_with_dot_cmds)
    segment_features["has_dot_in_middle"] = set_feature_intersection_count(segment_df, "cmd", has_dot_in_middle)
    # command types features

    return segment_features
