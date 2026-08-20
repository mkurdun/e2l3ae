import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from _datasets.utils import (
    Dataset,
    TimeBasedEvaluation,
    get_sparse_matrix_from_dataframe,
    preproces_html,
)


def NMSE(x, y):
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    return keras.losses.mean_squared_error(x, y)


def get_first_item(d):
    return d[next(iter(d.keys()))]
