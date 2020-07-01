"""
Codes for preprocessing the real-world datasets
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings

from preprocess.preprocessor import preprocess_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '-d', nargs='*', type=str, required=True, choices=['coat', 'yahoo'])

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    for data in args.datasets:
        preprocess_dataset(data=data)

        print('\n', '=' * 25, '\n')
        print(f'Finished Preprocessing {data}!')
        print('\n', '=' * 25, '\n')
