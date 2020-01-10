"""
Codes for preprocessing datasets used in the real-world experiments
in the paper "Unbiased Pairwise Learning from Implicit Feedback".
"""
import codecs
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.model_selection import train_test_split


class RealWorldLoader:
    """Load and Preprocess datasets."""

    def __init__(self) -> None:
        """Initialize Class."""

    def load(self, data: str, threshold: int = 4) -> Tuple:
        """Load and Preprocess datasets."""
        # load dataset.
        with codecs.open(f'../data/yahoo/train.txt', 'r', 'utf-8', errors='ignore') as f:
            data_train = pd.read_csv(f, delimiter='\t', header=None)
            data_train.rename(
                columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
            data_train.rate = np.array(
                data_train.rate.values >= threshold, dtype=bool)
        with codecs.open(f'../data/yahoo/test.txt', 'r', 'utf-8', errors='ignore') as f:
            data_test = pd.read_csv(f, delimiter='\t', header=None)
            data_test.rename(
                columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
            data_test.rate = np.array(
                data_test.rate.values >= threshold, dtype=bool)
        for _data in [data_train, data_test]:
            _data.user, _data.item = _data.user - 1, _data.item - 1
        # count the num. of users and items.
        num_users = data_train.user.max() + 1
        num_items = data_train.item.max() + 1
        # train-val-test, split
        train, test = data_train.values, data_test.values
        train, val = train_test_split(train, test_size=0.1, random_state=12345)
        # estimate propensity (relative item popularity)
        _, item_freq = np.unique(
            train[train[:, 2] == 1, 1], return_counts=True)
        prop = (item_freq / item_freq.max()) ** 0.5
        # only positive data
        train = train[train[:, 2] == 1, :2]
        val = val[val[:, 2] == 1, :2]

        return train, val, test, prop, num_users, num_items


def generate_real_data(data: str = 'yahoo', threshold: int = 4) -> None:
    """Generate real-world data."""
    np.random.seed(12345)
    loader = RealWorldLoader()
    train, val, test, prop, num_users, num_items = \
        loader.load(data=data, threshold=threshold)
    # creating training data
    all_data = pd.DataFrame(
        np.zeros((num_users, num_items))).stack().reset_index()
    all_data = all_data.values[:, :2]
    unlabeled_data = np.array(
        list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    train = np.r_[np.c_[train, np.ones(train.shape[0])],
                  np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]
    # for pointwise and pairwise approaches
    os.makedirs(f'../data/{data}/point/', exist_ok=True)
    os.makedirs(f'../data/{data}/test/', exist_ok=True)
    os.makedirs(f'../data/{data}/bpr/', exist_ok=True)
    os.makedirs(f'../data/{data}/ubpr/', exist_ok=True)
    np.save(file=f'../data/{data}/point/train.npy', arr=train.astype(np.int))
    np.save(file=f'../data/{data}/point/val.npy', arr=val.astype(np.int))
    np.save(file=f'../data/{data}/point/test.npy', arr=test.astype(np.int))
    np.save(file=f'../data/{data}/point/prop.npy', arr=prop)
    # pairwise
    # creating train data for pointwise recommenders.
    samples = 10
    train_pair = np.c_[train, prop[train[:, 1].astype(np.int)]]
    bpr_train = _bpr(data=train, n_samples=samples)
    ubpr_train = _ubpr(data=train_pair, n_samples=samples)
    np.save(file=f'../data/{data}/bpr/train.npy', arr=bpr_train)
    np.save(file=f'../data/{data}/ubpr/train.npy', arr=ubpr_train)


def _bpr(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'rating'])
    positive = df.query("rating == 1")
    negative = df.query("rating == 0")
    ret = positive.merge(negative, on="user") \
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"]) \
        .head(n_samples)

    # specifications for the columns of the generated data array
    # ==========================================================
    # columns |     0    |          1         |         2         |
    # factors |  user_id |  clicked_item_id   | unclicked_item_id |

    return ret[['user', 'item_x', 'item_y']].values


def _ubpr(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the unbiased bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'rating', 'theta'])
    positive = df.query("rating == 1")
    # negative = df.query("rating == 0")
    ret = positive.merge(df, on="user") \
        .sample(frac=1, random_state=12345) \
        .groupby(["user", "item_x"]) \
        .head(n_samples)
    ret = ret[ret["item_x"] != ret["item_y"]]

    # specifications for the columns of the generated data array
    # ==========================================================
    # columns |     0    |          1         |        2        |               3                 |           4           |           5           |
    # factors |  user_id |  clicked_item_id   | another_item_id | click_indicator_of_another_item | theta_of_clicked_item | theta_of_another_item |

    return ret[['user', 'item_x', 'item_y', 'rating_y', 'theta_x', 'theta_y']].values
