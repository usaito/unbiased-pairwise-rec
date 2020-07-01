"""

Exposure Matrix Factorization for collaborative filtering

CREATED: 2015-05-28 01:16:56 by Dawen Liang <dliang@ee.columbia.edu>

Reference
----------
https://github.com/dawenl/expo-mf
"""


from scipy import sparse
import bottleneck as bn
import os
import sys
import time
from math import sqrt

import numpy as np
from joblib import Parallel, delayed
from numpy import linalg as LA
from sklearn.base import BaseEstimator, TransformerMixin

floatX = np.float32
EPS = 1e-8


class ExpoMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, max_iter=10, batch_size=1000,
                 init_std=0.01, n_jobs=8, random_state=None, save_params=False,
                 save_dir='.', early_stopping=False, verbose=False, **kwargs):
        '''
        Exposure matrix factorization

        Parameters
        ---------
        n_components : int
            Number of latent factors
        max_iter : int
            Maximal number of iterations to perform
        batch_size: int
            Batch size to perform parallel update
        init_std: float
            The latent factors will be initialized as Normal(0, init_std**2)
        n_jobs: int
            Number of parallel jobs to update latent factors
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        save_params: bool
            Whether to save parameters after each iteration
        save_dir: str
            The directory to save the parameters
        early_stopping: bool
            Whether to early stop the training by monitoring performance on
            validation set
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init_std = init_std
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.verbose = verbose

        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.seed(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        ''' Model hyperparameters

        Parameters
        ---------
        lambda_theta, lambda_beta: float
            Regularization parameter for user (lambda_theta) and item factors (
            lambda_beta). Default value 1e-5. Since in implicit feedback all
            the n_users-by-n_items data points are used for training,
            overfitting is almost never an issue
        lambda_y: float
            inverse variance on the observational model. Default value 1.0
        init_mu: float
            All the \mu_{i} will be initialized as init_mu. Default value is
            0.01. This number should change according to the sparsity of the
            data (sparser data with smaller init_mu). In the experiment, we
            select the value from validation set
        a, b: float
            Prior on \mu_{i} ~ Beta(a, b)
        '''
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_y = float(kwargs.get('lam_y', 1.0))
        self.init_mu = float(kwargs.get('init_mu', 0.01))
        self.a = float(kwargs.get('a', 1.0))
        self.b = float(kwargs.get('b', 1.0))

    def _init_params(self, n_users, n_items):
        ''' Initialize all the latent factors '''
        self.theta = self.init_std * \
            np.random.randn(n_users, self.n_components).astype(floatX)
        self.beta = self.init_std * \
            np.random.randn(n_items, self.n_components).astype(floatX)
        self.mu = self.init_mu * np.ones(n_items, dtype=floatX)

    def fit(self, X, vad_data=None, **kwargs):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_users, n_items)
            Training data.

        vad_data: scipy.sparse.csr_matrix, shape (n_users, n_items)
            Validation data.

        **kwargs: dict
            Additional keywords to evaluation function call on validation data

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_users, n_items = X.shape
        self._init_params(n_users, n_items)
        self._update(X, vad_data, **kwargs)
        return self

    def transform(self, X):
        pass

    def _update(self, X, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        n_users = X.shape[0]
        XT = X.T.tocsr()  # pre-compute this
        self.vad_ndcg = -np.inf
        for i in range(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(X, XT)
            self._update_expo(X, n_users)
            if vad_data is not None:
                vad_ndcg = self._validate(X, vad_data, **kwargs)
                if self.early_stopping and self.vad_ndcg > vad_ndcg:
                    break  # we will not save the parameter for this iteration
                self.vad_ndcg = vad_ndcg

            if self.save_params:
                self._save_params(i)
        pass

    def _update_factors(self, X, XT):
        '''Update user and item collaborative factors with ALS'''
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user factors...')
        self.theta = recompute_factors(self.beta, self.theta, X,
                                       self.lam_theta / self.lam_y,
                                       self.lam_y,
                                       self.mu,
                                       self.n_jobs,
                                       batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating item factors...')
        self.beta = recompute_factors(self.theta, self.beta, XT,
                                      self.lam_beta / self.lam_y,
                                      self.lam_y,
                                      self.mu,
                                      self.n_jobs,
                                      batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating item factors: time=%.2f'
                  % (time.time() - start_t))
            sys.stdout.flush()
        pass

    def _update_expo(self, X, n_users):
        '''Update exposure prior'''
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating exposure prior...')

        start_idx = np.arange(0, n_users, self.batch_size).tolist()
        end_idx = start_idx[1:] + [n_users]

        A_sum = np.zeros_like(self.mu)
        for lo, hi in zip(start_idx, end_idx):
            A_sum += a_row_batch(X[lo:hi], self.theta[lo:hi], self.beta,
                                 self.lam_y, self.mu).sum(axis=0)
        self.mu = (self.a + A_sum - 1) / (self.a + self.b + n_users - 2)
        if self.verbose:
            print('\r\tUpdating exposure prior: time=%.2f'
                  % (time.time() - start_t))
            sys.stdout.flush()
        pass

    def _validate(self, X, vad_data, **kwargs):
        '''Compute validation metric (NDCG@k)'''
        vad_ndcg = normalized_dcg_at_k(
            X, vad_data, self.theta, self.beta, **kwargs)
        if self.verbose:
            print('\tValidation NDCG@k: %.4f' % vad_ndcg)
            sys.stdout.flush()
        return vad_ndcg

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'ExpoMF_K%d_mu%.1e_iter%d.npz' % (self.n_components,
                                                     self.init_mu, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta,
                 V=self.beta, mu=self.mu)


# Utility functions #

def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def a_row_batch(Y_batch, theta_batch, beta, lam_y, mu):
    '''Compute the posterior of exposure latent variables A by batch'''
    pEX = sqrt(lam_y / 2 * np.pi) * \
        np.exp(-lam_y * theta_batch.dot(beta.T)**2 / 2)
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[Y_batch.nonzero()] = 1.
    return A


def _solve(k, A_k, X, Y, f, lam, lam_y, mu):
    '''Update one single factor'''
    s_u, i_u = get_row(Y, k)
    a = np.dot(s_u * A_k[i_u], X[i_u])
    B = X.T.dot(A_k[:, np.newaxis] * X) + lam * np.eye(f)
    return LA.solve(B, a)


def _solve_batch(lo, hi, X, X_old_batch, Y, m, f, lam, lam_y, mu):
    '''Update factors by batch, will eventually call _solve() on each factor to
    keep the parallel process busy'''
    assert X_old_batch.shape[0] == hi - lo

    if mu.size == X.shape[0]:  # update users
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu)
    else:  # update items
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu[lo:hi,
                                                                  np.newaxis])

    X_batch = np.empty_like(X_old_batch, dtype=X_old_batch.dtype)
    for ib, k in enumerate(np.arange(lo, hi)):
        X_batch[ib] = _solve(k, A_batch[ib], X, Y, f, lam, lam_y, mu)
    return X_batch


def recompute_factors(X, X_old, Y, lam, lam_y, mu, n_jobs, batch_size=1000):
    '''Regress X to Y with exposure matrix (computed on-the-fly with X_old) and
    ridge term lam by embarrassingly parallelization. All the comments below
    are in the view of computing user factors'''
    m, n = Y.shape  # m = number of users, n = number of items
    assert X.shape[0] == n
    assert X_old.shape[0] == m
    f = X.shape[1]  # f = number of factors

    start_idx = np.arange(0, m, batch_size).tolist()
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, X, X_old[lo:hi], Y, m, f, lam, lam_y, mu)
        for lo, hi in zip(start_idx, end_idx))
    X_new = np.vstack(res)
    return X_new


def prec_at_k(train_data, heldout_data, U, V, batch_users=5000, k=20,
              mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(precision_at_k_batch(train_data, heldout_data,
                                        U, V.T, user_idx, k=k,
                                        mu=mu, vad_data=vad_data))
    mn_prec = np.hstack(res)
    if callable(agg):
        return agg(mn_prec)
    return mn_prec


def recall_at_k(train_data, heldout_data, U, V, batch_users=5000, k=20,
                mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(recall_at_k_batch(train_data, heldout_data,
                                     U, V.T, user_idx, k=k,
                                     mu=mu, vad_data=vad_data))
    mn_recall = np.hstack(res)
    if callable(agg):
        return agg(mn_recall)
    return mn_recall


def ric_rank_at_k(train_data, heldout_data, U, V, batch_users=5000, k=5,
                  mu=None, vad_data=None):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(mean_rrank_at_k_batch(train_data, heldout_data,
                                         U, V.T, user_idx, k=k,
                                         mu=mu, vad_data=vad_data))
    mrrank = np.hstack(res)
    return mrrank[mrrank > 0].mean()


def mean_perc_rank(train_data, heldout_data, U, V, batch_users=5000,
                   mu=None, vad_data=None):
    n_users = train_data.shape[0]
    mpr = 0
    for user_idx in user_idx_generator(n_users, batch_users):
        mpr += mean_perc_rank_batch(train_data, heldout_data, U, V.T, user_idx,
                                    mu=mu, vad_data=vad_data)
    mpr /= heldout_data.sum()
    return mpr


def normalized_dcg(train_data, heldout_data, U, V, batch_users=5000,
                   mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_binary_batch(train_data, heldout_data, U, V.T,
                                     user_idx, mu=mu, vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def normalized_dcg_at_k(train_data, heldout_data, U, V, batch_users=5000,
                        k=100, mu=None, vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_binary_at_k_batch(train_data, heldout_data, U, V.T,
                                          user_idx, k=k, mu=mu,
                                          vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def map_at_k(train_data, heldout_data, U, V, batch_users=5000, k=100, mu=None,
             vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(MAP_at_k_batch(train_data, heldout_data, U, V.T, user_idx,
                                  k=k, mu=mu, vad_data=vad_data))
    map = np.hstack(res)
    if callable(agg):
        return agg(map)
    return map


# helper functions #

def user_idx_generator(n_users, batch_users):
    ''' helper function to generate the user index to loop through the dataset
    '''
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


def _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=None,
                     vad_data=None):
    n_songs = train_data.shape[1]
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_songs), dtype=bool)
    item_idx[train_data[user_idx].nonzero()] = True
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = True
    X_pred = Et[user_idx].dot(Eb)
    if mu is not None:
        if isinstance(mu, np.ndarray):
            assert mu.size == n_songs  # mu_i
            X_pred *= mu
        elif isinstance(mu, dict):  # func(mu_ui)
            params, func = mu['params'], mu['func']
            args = [params[0][user_idx], params[1]]
            if len(params) > 2:  # for bias term in document or length-scale
                args += [params[2][user_idx]]
            if not callable(func):
                raise TypeError("expecting a callable function")
            X_pred *= func(*args)
        else:
            raise ValueError("unsupported mu type")
    X_pred[item_idx] = -np.inf
    return X_pred


def precision_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                         k=20, normalize=True, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx = bn.argpartsort(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    if normalize:
        precision = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    else:
        precision = tmp / k
    return precision


def recall_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                      k=20, normalize=True, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx = bn.argpartsort(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def mean_rrank_at_k_batch(train_data, heldout_data, Et, Eb,
                          user_idx, k=5, mu=None, vad_data=None):
    '''
    mean reciprocal rank@k: For each user, make predictions and rank for
    all the items. Then calculate the mean reciprocal rank for the top K that
    are in the held-out set.
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rrank = 1. / (np.argsort(np.argsort(-X_pred, axis=1), axis=1) + 1)
    X_true_binary = (heldout_data[user_idx] > 0).toarray()

    heldout_rrank = X_true_binary * all_rrank
    top_k = bn.partsort(-heldout_rrank, k, axis=1)
    return -top_k[:, :k].mean(axis=1)


def NDCG_binary_batch(train_data, heldout_data, Et, Eb, user_idx,
                      mu=None, vad_data=None):
    '''
    normalized discounted cumulative gain for binary relevance
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1)
    # build the discount template
    tp = 1. / np.log2(np.arange(2, n_items + 2))
    all_disc = tp[all_rank]

    X_true_binary = (heldout_data[user_idx] > 0).tocoo()
    disc = sparse.csr_matrix((all_disc[X_true_binary.row, X_true_binary.col],
                              (X_true_binary.row, X_true_binary.col)),
                             shape=all_disc.shape)
    DCG = np.array(disc.sum(axis=1)).ravel()
    IDCG = np.array([tp[:n].sum()
                     for n in heldout_data[user_idx].getnnz(axis=1)])
    return DCG / IDCG


def NDCG_binary_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                           mu=None, k=100, vad_data=None):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx_topk_part = bn.argpartsort(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    heldout_batch = heldout_data[user_idx]
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def MAP_at_k_batch(train_data, heldout_data, Et, Eb, user_idx, mu=None, k=100,
                   vad_data=None):
    '''
    mean average precision@k
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=mu,
                              vad_data=vad_data)
    idx_topk_part = bn.argpartsort(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    aps = np.zeros(batch_users)
    for i, idx in enumerate(range(user_idx.start, user_idx.stop)):
        actual = heldout_data[idx].nonzero()[1]
        if len(actual) > 0:
            predicted = idx_topk[i]
            aps[i] = apk(actual, predicted, k=k)
        else:
            aps[i] = np.nan
    return aps


def mean_perc_rank_batch(train_data, heldout_data, Et, Eb, user_idx,
                         mu=None, vad_data=None):
    '''
    mean percentile rank for a batch of users
    MPR of the full set is the sum of batch MPR's divided by the sum of all the
    feedbacks. (Eq. 8 in Hu et al.)
    This metric not necessarily constrains the data to be binary
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx, batch_users,
                              mu=mu, vad_data=vad_data)
    all_perc = np.argsort(np.argsort(-X_pred, axis=1), axis=1) / \
        np.isfinite(X_pred).sum(axis=1, keepdims=True).astype(np.float32)
    perc_batch = (all_perc[heldout_data[user_idx].nonzero()] *
                  heldout_data[user_idx].data).sum()
    return perc_batch


# steal from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=100):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual:  # and p not in predicted[:i]: # not necessary for us since we will not make duplicated recs
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # we handle this part before making the function call
    # if not actual:
    #    return np.nan

    return score / min(len(actual), k)
