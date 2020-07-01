"""
Recommender models used for the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def create_losses(self) -> None:
        """Create the losses."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()


@dataclass
class PointwiseRecommender(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""
    num_users: np.array
    num_items: np.array
    dim: int
    lam: float
    eta: float
    weight: float = 1.
    clip: float = 0.

    def __post_init__(self,) -> None:
        """Initialize Class."""
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_ph')
        self.items = tf.placeholder(tf.int32, [None], name='item_ph')
        self.scores = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.labels = tf.placeholder(tf.float32, [None, 1], name='label_ph')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            self.user_embeddings = tf.get_variable('user_embeddings', shape=[self.num_users, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable('item_embeddings', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)

        with tf.variable_scope('prediction'):
            self.logits = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = tf.sigmoid(tf.expand_dims(self.logits, 1), name='sigmoid_prediction')

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the unbiased loss for the ideal loss function with binary implicit feedback.
            scores = tf.clip_by_value(self.scores, clip_value_min=self.clip, clip_value_max=1.0)
            local_losses = (self.labels / scores) * tf.square(1. - self.preds)
            local_losses += self.weight * (1 - self.labels / scores) * tf.square(self.preds)
            local_losses = tf.clip_by_value(local_losses, clip_value_min=-1000, clip_value_max=1000)
            numerator = tf.reduce_sum(self.labels + self.weight * (1 - self.labels))
            self.unbiased_loss = tf.reduce_sum(local_losses) / numerator

            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.unbiased_loss + self.lam * reg_embeds

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)


@dataclass
class PairwiseRecommender(AbstractRecommender):
    """Implicit Recommenders based on pairwise approach."""
    num_users: np.array
    num_items: np.array
    dim: int = 20
    lam: float = 1e-4
    eta: float = 0.005
    beta: float = 0.0

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_ph1')
        self.pos_items = tf.placeholder(tf.int32, [None], name='item_ph1')
        self.scores1 = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.items2 = tf.placeholder(tf.int32, [None], name='item_ph2')
        self.scores2 = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.labels2 = tf.placeholder(tf.float32, [None, 1], name='label_ph2')
        self.rel1 = tf.placeholder(tf.float32, [None, 1], name='rel_ph1')
        self.rel2 = tf.placeholder(tf.float32, [None, 1], name='rel_ph2')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            self.user_embeddings = tf.get_variable('user_embeddings', shape=[self.num_users, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable('item_embeddings', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_p_embed = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
            self.i_embed2 = tf.nn.embedding_lookup(self.item_embeddings, self.items2)

        with tf.variable_scope('prediction'):
            self.preds1 = tf.reduce_sum(tf.multiply(self.u_embed, self.i_p_embed), 1)
            self.preds2 = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed2), 1)
            self.preds = tf.sigmoid(tf.expand_dims(self.preds1 - self.preds2, 1))

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the naive pairwise loss.
            local_losses = - self.rel1 * (1 - self.rel2) * tf.log(self.preds)
            self.ideal_loss = tf.reduce_sum(local_losses) / tf.reduce_sum(self.rel1 * (1 - self.rel2))
            # define the unbiased pairwise loss.
            local_losses = - (1 / self.scores1) * (1 - (self.labels2 / self.scores2)) * tf.log(self.preds)
            # non-negative
            local_losses = tf.clip_by_value(local_losses, clip_value_min=-self.beta, clip_value_max=10e5)
            self.unbiased_loss = tf.reduce_mean(local_losses)

            reg_embeds = tf.nn.l2_loss(self.user_embeddings)
            reg_embeds += tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.unbiased_loss + self.lam * reg_embeds

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)
