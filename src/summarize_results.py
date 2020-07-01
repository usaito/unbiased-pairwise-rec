"""
Codes for summarizing results of the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd


# configurations.
all_models = ['wmf', 'expomf', 'relmf', 'bpr', 'ubpr']
K = [3, 5, 8]
metrics = ['DCG', 'Recall', 'MAP']
col_names = [f'{m}@{k}' for m in metrics for k in K]
rel_col_names = [f'{m}@5' for m in metrics]


def summarize_results(data: str, path: Path) -> None:
    """Load and save experimental results."""
    suffixes = ['all'] if data == 'coat' else ['cold-user', 'rare-item', 'all']
    for suffix in suffixes:
        aoa_list = []
        for model in all_models:
            file = f'../logs/{data}/{model}/results/aoa_{suffix}.csv'
            aoa_list.append(pd.read_csv(file, index_col=0).mean(1))
        results_df = pd.concat(aoa_list, 1).round(7).T
        results_df.index = all_models
        results_df[col_names].to_csv(path / f'ranking_{suffix}.csv')


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '-d', required=True, nargs='*', type=str, choices=['coat', 'yahoo'])

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    for data in args.datasets:
        path = Path(f'../paper_results/{data}')
        path.mkdir(parents=True, exist_ok=True)
        summarize_results(data=data, path=path)
