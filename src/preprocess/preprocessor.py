"""
Codes for preprocessing datasets used in the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import codecs
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


def transform_rating(ratings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Transform ratings into graded relevance information."""
    ratings -= 1
    return eps + (1. - eps) * (2 ** ratings - 1) / (2 ** np.max(ratings) - 1)


def preprocess_dataset(data: str):
    """Load and preprocess datasets."""
    np.random.seed(12345)
    if data == 'yahoo':
        cols = {0: 'user', 1: 'item', 2: 'rate'}
        with codecs.open(f'../data/yahoo/raw/train.txt', 'r', 'utf-8', errors='ignore') as f:
            train_ = pd.read_csv(f, delimiter='\t', header=None)
            train_.rename(columns=cols, inplace=True)
        with codecs.open(f'../data/yahoo/raw/test.txt', 'r', 'utf-8', errors='ignore') as f:
            test_ = pd.read_csv(f, delimiter='\t', header=None)
            test_.rename(columns=cols, inplace=True)
        for _data in [train_, test_]:
            _data.user, _data.item = _data.user - 1, _data.item - 1
    elif data == 'coat':
        cols = {'level_0': 'user', 'level_1': 'item', 2: 'rate', 0: 'rate'}
        with codecs.open(f'../data/coat/raw/train.ascii', 'r', 'utf-8', errors='ignore') as f:
            train_ = pd.read_csv(f, delimiter=' ', header=None)
            train_ = train_.stack().reset_index().rename(columns=cols)
            train_ = train_[train_.rate != 0].reset_index(drop=True)
        with codecs.open(f'../data/coat/raw/test.ascii', 'r', 'utf-8', errors='ignore') as f:
            test_ = pd.read_csv(f, delimiter=' ', header=None)
            test_ = test_.stack().reset_index().rename(columns=cols)
            test_ = test_[test_.rate != 0].reset_index(drop=True)
    # count the num. of users and items.
    num_users, num_items = train_.user.max() + 1, train_.item.max() + 1
    train, test = train_.values, test_.values
    # transform rating into (0,1)-scale.
    test[:, 2] = transform_rating(ratings=test[:, 2], eps=0.0)
    rel_train = np.random.binomial(n=1, p=transform_rating(ratings=train[:, 2], eps=0.1))

    # exstract only positive (relevant) user-item pairs
    train = train[rel_train == 1, :2]
    # creating training data
    all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index().values[:, :2]
    unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    train = np.r_[np.c_[train, np.ones(train.shape[0])], np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]
    # estimate propensities and user-item frequencies.
    if data == 'yahoo':
        user_freq = np.unique(train[train[:, 2] == 1, 0], return_counts=True)[1]
        item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)[1]
        pscore = (item_freq / item_freq.max()) ** 0.5
    elif data == 'coat':
        matrix = sparse.lil_matrix((num_users, num_items))
        for (u, i) in train[:, :2]:
            matrix[u, i] = 1
        pscore = np.clip(np.array(matrix.mean(axis=0)).flatten() ** 0.5, a_max=1.0, a_min=1e-6)
    # train-val split using the raw training datasets
    train, val = train_test_split(train, test_size=0.1, random_state=12345)
    # save preprocessed datasets
    path_data = Path(f'../data/{data}')
    (path_data / 'point').mkdir(parents=True, exist_ok=True)
    (path_data / 'pair').mkdir(parents=True, exist_ok=True)
    # pointwise
    np.save(file=path_data / 'point/train.npy', arr=train.astype(np.int))
    np.save(file=path_data / 'point/val.npy', arr=val.astype(np.int))
    np.save(file=path_data / 'point/test.npy', arr=test)
    np.save(file=path_data / 'point/pscore.npy', arr=pscore)
    if data == 'yahoo':
        np.save(file=path_data / 'point/user_freq.npy', arr=user_freq)
        np.save(file=path_data / 'point/item_freq.npy', arr=item_freq)
    # pairwise
    samples = 10
    bpr_train = _bpr(data=train, n_samples=samples)
    ubpr_train = _ubpr(data=train, pscore=pscore, n_samples=samples)
    bpr_val = _bpr(data=val, n_samples=samples)
    ubpr_val = _ubpr(data=val, pscore=pscore, n_samples=samples)
    pair_test = _bpr_test(data=test, n_samples=samples)
    np.save(file=path_data / 'pair/bpr_train.npy', arr=bpr_train)
    np.save(file=path_data / 'pair/ubpr_train.npy', arr=ubpr_train)
    np.save(file=path_data / 'pair/bpr_val.npy', arr=bpr_val)
    np.save(file=path_data / 'pair/ubpr_val.npy', arr=ubpr_val)
    np.save(file=path_data / 'pair/test.npy', arr=pair_test)


def _bpr(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'click'])
    positive = df.query("click == 1")
    negative = df.query("click == 0")
    ret = positive.merge(negative, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y']].values


def _bpr_test(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'gamma'])
    ret = df.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y', 'gamma_x', 'gamma_y']].values


def _ubpr(data: np.ndarray, pscore: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the unbiased bpr."""
    data = np.c_[data, pscore[data[:, 1].astype(int)]]
    df = pd.DataFrame(data, columns=['user', 'item', 'click', 'theta'])
    positive = df.query("click == 1")
    ret = positive.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)
    ret = ret[ret["item_x"] != ret["item_y"]]

    return ret[['user', 'item_x', 'item_y', 'click_y', 'theta_x', 'theta_y']].values
