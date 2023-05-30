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
        5.  V - has zip/compression cmds
        6.  V - has encryption cmds
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

    # boolean command counter
    # segment_cmd_set = set(segment_df["cmd"].tolist())
    # for global_cmd in global_cmds:
    #     segment_features[f"is_in_segment{global_cmd}"] = 1 if global_cmd in segment_cmd_set else 0

    # features
    segment_features["cmd_most_used"] = next(iter(segment_cmd_value_counts))
    segment_features["first_cmd"] = segment_df["cmd"].iloc[0]
    segment_features["last_cmd"] = segment_df["cmd"].iloc[SEGMENT_LEN - 1]
    segment_features["unique_cmds"] = len(segment_df["cmd"].unique())

    segment_features["max_cmd_len"] = segment_df["cmd"].apply(lambda s: len(s)).max()
    segment_features["min_cmd_len"] = segment_df["cmd"].apply(lambda s: len(s)).min()
    segment_features["mean_cmd_length"] = segment_df["cmd"].apply(lambda s: len(s)).mean()

    # longest subsequence of same commands
    s = segment_df["cmd_code"].diff().astype(bool)
    segment_features["longest_same_cmd_sequence"] = (~s).cumsum()[
        s].value_counts().max()  # TODO - there is a bug here, getting 100s

    get_features = lambda group: set_feature_intersection_count(segment_df, "cmd", group)
    get_features_not_in_train = lambda group: set_feature_intersection_count(
        segment_df[segment_df["cmd"].isin(user_cmd_set_not_in_train)], "cmd", group)


    # does command appear boolean features


    #   command types features
    for feature_name, feature_cmd_list in cmd_features_groups.items():
        segment_features[feature_name] = get_features(feature_cmd_list)

    #   commands not in train
    segment_features["cmds_not_in_train"] = get_features(user_cmd_set_not_in_train)

    for feature_name, feature_cmd_list in cmd_features_groups.items():
        segment_features[f"not_in_train_{feature_name}"] = get_features_not_in_train(feature_cmd_list)

    return segment_features
