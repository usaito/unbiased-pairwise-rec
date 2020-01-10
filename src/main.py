"""
Codes for running the real-world experiments
in the paper "Unbiased Pairwise Learning from Implicit Feedback".
"""

import argparse
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from data_generator import generate_real_data
from trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, choices=[
                    'bpr', 'ubpr', 'relmf'], help='a used model')
parser.add_argument('data', type=str, choices=['yahoo'], help='a used dataset')
parser.add_argument('--threshold', default=4, type=int,
                    help='a threshold for binalize rating')
parser.add_argument('--eta', default=5e-3, type=float,
                    help='learning_rate for Adam')
parser.add_argument('--batch_size', default=15, type=int,
                    help='batch_size for mini-batch sampling')
parser.add_argument('--max_iters', default=301, type=int,
                    help='maximun num of iterations for Adam')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # hyper-parameters
    data = args.data
    eta = args.eta
    batch_size = args.batch_size
    max_iters = args.max_iters
    threshold = args.threshold
    model_name = args.model_name

    # run simulations
    mlflow.set_experiment(f'real-{data}')
    with mlflow.start_run() as run:
        # pre-process datasets for learning pointwise&pairwise recommenders
        generate_real_data(data=data, threshold=threshold)

        print('\n', '=' * 25, '\n')
        print(f'Finished generating training data for pairwise learning!')
        print('\n', '=' * 25, '\n')

        # start training recommender
        trainer = Trainer(
            data=data, batch_size=batch_size, max_iters=max_iters, eta=eta, model_name=model_name)
        trainer.run()

        # aggregate results
        results_df = pd.read_csv(
            f'../logs/{model_name}/{data}/results/aoa_results_all.csv', index_col=0)
        mlflow.log_metric('MAP5', results_df.loc['MAP@5', model_name])
        mlflow.log_metric('Recall5', results_df.loc['Recall@5', model_name])
        mlflow.log_metric('DCG5', results_df.loc['DCG@5', model_name])

        print('\n', '=' * 25, '\n')
        print(f'Finished Running {model_name}!')
        print('\n', '=' * 25, '\n')

        mlflow.log_param('data', data)
        mlflow.log_param('threshold', threshold)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('max_iters', max_iters)
        mlflow.log_param('model_name', model_name)

        mlflow.log_artifacts(f'../logs/{model_name}/{data}/results/')
