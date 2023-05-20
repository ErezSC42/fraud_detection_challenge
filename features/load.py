import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from constants import *


def user_list_to_df(user_id: str, user_data_list: List[str]):
    df_user = pd.DataFrame({
        "cmd": user_data_list
    })
    df_user["user"] = user_id
    df_user["split"] = "train"
    df_user["segment_id"] = np.repeat(range(0, int(len(user_data_list) / SEGMENT_LEN)), SEGMENT_LEN)
    df_user["cmd"] = df_user["cmd"].astype("category")
    return df_user


def load_user_data(user_id: str, file_path: os.PathLike) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(file_path, "r") as fp:
        user_data = fp.readlines()
        user_data = [s.strip() for s in user_data]

    # get training data:
    train_user_data = user_data[:TRAIN_HEADER_COUNT]
    test_user_data = user_data[TRAIN_HEADER_COUNT:]

    # convert to dataframes
    train_segments = user_list_to_df(user_id, train_user_data)
    test_segments = user_list_to_df(user_id, test_user_data)

    return train_segments, test_segments
