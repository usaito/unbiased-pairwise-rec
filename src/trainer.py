"""
Codes for training recommenders used in the real-world experiments
in the paper "Unbiased Pairwise Learning from Implicit Feedback".
"""
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

from evaluator import AverageOverAllEvaluator, UnbiasedEvaluator, evaluate
from model import PairwiseRecommender, PointwiseRecommender


def pointwise_trainer(
        sess: tf.Session, data: str, model: PointwiseRecommender,
        train: np.ndarray, test: np.ndarray, propensity: np.ndarray,
        max_iters: int = 1000, batch_size: int = 2**12, model_name: str = 'relmf') -> None:
    """Train and evaluate implicit pointwise recommender."""
    train_loss_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # propensity for train
    propensity = propensity[train[:, 1].astype(np.int)]
    # positive and unlabeled data for training set
    pos_train = train[train[:, 2] == 1]
    prop_pos_train = propensity[train[:, 2] == 1]
    num_pos = np.sum(train[:, 2])
    unlabeled_train = train[train[:, 2] == 0]
    prop_unlabeled_train = propensity[train[:, 2] == 0]
    num_unlabeled = np.sum(1 - train[:, 2])
    # train the given implicit recommender
    np.random.seed(12345)
    for i in np.arange(max_iters):
        # positive mini-batch sampling
        # the same num. of postive and negative samples are used in each batch
        # for the pointwise training
        pos_idx = np.random.choice(
            np.arange(num_pos, dtype=int), size=np.int(batch_size / 2))
        unlabeled_idx = np.random.choice(
            np.arange(num_unlabeled, dtype=int), size=np.int(batch_size / 2))
        # mini-batch samples
        train_batch = np.r_[pos_train[pos_idx], unlabeled_train[unlabeled_idx]]
        train_label = train_batch[:, 2]
        # define propensity score
        prop_ = np.r_[prop_pos_train[pos_idx],
                      prop_unlabeled_train[unlabeled_idx]]
        train_score = np.expand_dims(prop_, 1)

        # update user-item latent factors and calculate training loss
        _, loss = sess.run(
            [model.apply_grads, model.weighted_mse],
            feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                       model.labels: np.expand_dims(train_label, 1), model.scores: train_score})
        train_loss_list.append(loss)

    # save embeddings.
    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
    np.save(file=f'../logs/{model_name}/embeds/user_embed.npy', arr=u_emb)
    np.save(file=f'../logs/{model_name}/embeds/item_embed.npy', arr=i_emb)
    # save train and val loss curves.
    np.save(file=f'../logs/{model_name}/loss/train.npy',
            arr=np.array(train_loss_list))

    sess.close()


def pairwise_trainer(
        sess: tf.Session, data: str, model: PairwiseRecommender,
        train: np.ndarray, train_point: np.ndarray, test_point: np.ndarray,
        max_iters: int = 1000, batch_size: int = 2**12, model_name: str = 'bpr') -> None:
    """Train and evaluate implicit pairwise recommenders."""
    train_loss_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # count the num of training data.
    num_train = train.shape[0]
    # train the given implicit recommender
    np.random.seed(12345)
    for i in np.arange(max_iters):
        # mini-batch samples
        idx = np.random.choice(
            np.arange(num_train, dtype=int), size=np.int(batch_size))
        train_batch = train[idx]

        # update user-item latent factors
        if model_name in 'bpr':
            scores = np.ones((batch_size, 1))
            labels2 = np.zeros((batch_size, 1))
            _, loss = sess.run(
                [model.apply_grads, model.loss],
                feed_dict={model.users: train_batch[:, 0],
                           model.pos_items: train_batch[:, 1],
                           model.scores1: scores,
                           model.items2: train_batch[:, 2],
                           model.labels2: labels2,
                           model.scores2: scores})
        elif 'ubpr' in model_name:
            _, loss = sess.run(
                [model.apply_grads, model.loss],
                feed_dict={model.users: train_batch[:, 0],
                           model.pos_items: train_batch[:, 1],
                           model.scores1: np.expand_dims(train_batch[:, 4], 1),
                           model.items2: train_batch[:, 2],
                           model.labels2: np.expand_dims(train_batch[:, 3], 1),
                           model.scores2: np.expand_dims(train_batch[:, 5], 1)})
        train_loss_list.append(loss)

    # save embeddings.
    u_emb, i_emb = sess.run([model.user_embeddings, model.item_embeddings])
    np.save(file=f'../logs/{model_name}/embeds/user_embed.npy', arr=u_emb)
    np.save(file=f'../logs/{model_name}/embeds/item_embed.npy', arr=i_emb)
    # save train and val loss curves.
    np.save(file=f'../logs/{model_name}/loss/train.npy',
            arr=np.array(train_loss_list))

    sess.close()


class Trainer:
    """Trainer Class for ImplicitRecommender."""

    def __init__(self, data: str, max_iters: int = 1000, batch_size: int = 12,
                 eta: float = 0.1, model_name: str = 'bpr') -> None:
        """Initialize class."""
        self.data = data
        best_params = json.loads(
            json.load(open(f'../logs/{model_name}/{self.data}/tuning/best_params.json', 'r')))
        self.dim = np.int(best_params['dim'])
        self.lam = best_params['lam']
        self.clip = best_params['clip'] if model_name == 'relmf' else 0.
        self.beta = best_params['beta'] if model_name == 'ubpr' else 0.
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name
        # make directory
        os.makedirs(f'../logs/{model_name}/embeds/', exist_ok=True)
        os.makedirs(f'../logs/{model_name}/loss/', exist_ok=True)
        os.makedirs(
            f'../logs/{model_name}/{self.data}/results/', exist_ok=True)

    def run(self) -> None:
        """Train implicit recommenders."""
        train_point = np.load(f'../data/{self.data}/point/train.npy')
        val_point = np.load(f'../data/{self.data}/point/val.npy')
        test_point = np.load(f'../data/{self.data}/point/test.npy')
        prop = np.load(f'../data/{self.data}/point/prop.npy')
        num_users = np.int(train_point[:, 0].max() + 1)
        num_items = np.int(train_point[:, 1].max() + 1)
        if self.model_name in ['bpr', 'ubpr']:
            train = np.load(f'../data/{self.data}/{self.model_name}/train.npy')

        tf.set_random_seed(12345)
        ops.reset_default_graph()
        sess = tf.Session()
        if self.model_name in ['ubpr', 'bpr']:
            pair = PairwiseRecommender(
                num_users=num_users, num_items=num_items, dim=self.dim,
                lam=self.lam, eta=self.eta, beta=self.beta)
            pairwise_trainer(
                sess, data=self.data, model=pair, train=train,
                train_point=train_point, test_point=test_point,
                max_iters=self.max_iters, batch_size=2**self.batch_size, model_name=self.model_name)

        elif self.model_name == 'relmf':
            point = PointwiseRecommender(
                num_users=num_users, num_items=num_items,
                clip=self.clip, dim=self.dim, lam=self.lam, eta=self.eta)
            pointwise_trainer(
                sess, data=self.data, model=point,
                train=train_point, test=test_point, propensity=prop,
                max_iters=self.max_iters, batch_size=2**self.batch_size, model_name=self.model_name)

        evaluate(
            data=self.data, train=train_point, val=val_point, test=test_point,
            propensity=prop, model_name=self.model_name, rare=500, k=[1, 3, 5])
