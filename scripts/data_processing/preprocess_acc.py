import os
import argparse

from utils import preprocess_df


def preprocess_save_df(args):
    # load dataset and preprocess to include method and setting columns, rename val_acc to acc
    # drop columns
    # filter
    df = preprocess_df(
        args.input_file,
        'acc',
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
    )

    # aggregate and save main results
    fp = os.path.join(args.results_dir, args.output_file)
    df.to_csv(fp, header=True, index=False)

    return df


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('data', 'saw_stage2.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=None)

    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str, default=None)
    parser.add_argument('--filter_serials', nargs='+', type=int, default=None)

    # output
    parser.add_argument('--output_file', type=str, default='saw_acc_preprocessed.csv',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'acc'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df = preprocess_save_df(args)
    return df


if __name__ == '__main__':
    main()
