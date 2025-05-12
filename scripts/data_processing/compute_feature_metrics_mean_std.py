import os
import re
import argparse
import pandas as pd

from utils import add_all_cols_group


def get_cols(cols, pattern='cka', split=False):
    test = True if split == 'test' else False

    if test:
        if pattern == 'cka':
            pattern = r'^cka_\d+_test$'
        elif pattern == 'cka_change_mean':
            pattern = r'^cka_change_mean_\d+_test$'
        elif pattern == 'cka_change2_mean':
            pattern = r'^cka_change2_mean_\d+_test$'
        elif pattern == 'dist':
            pattern = r'^dist_\d+_test$'
        elif pattern == 'dist_norm':
            pattern = r'^dist_norm_\d+_test$'
        elif pattern == 'l2':
            pattern = r'^l2_norm_\d+_test$'
        elif pattern == 'attn_mean':
            pattern = r'^attn_mean_\d+_test$'
        elif pattern == 'attn_std':
            pattern = r'^attn_mean_\d+_test$'

    else:
        if pattern == 'cka':
            pattern = r'^cka_\d+_train$'
        elif pattern == 'cka_change_mean':
            pattern = r'^cka_change_mean_\d+_train$'
        elif pattern == 'cka_change2_mean':
            pattern = r'^cka_change2_mean_\d+_train$'
        elif pattern == 'dist':
            pattern = r'^dist_\d+_train$'
        elif pattern == 'dist_norm':
            pattern = r'^dist_norm_\d+_train$'
        elif pattern == 'l2':
            pattern = r'^l2_norm_\d+_train$'
        elif pattern == 'attn_mean':
            pattern = r'^attn_mean_\d+_train$'
        elif pattern == 'attn_std':
            pattern = r'^attn_mean_\d+_train$'

    cols = [col for col in cols if re.match(pattern, col)]
    return cols


def process_df(input_file, output_file, include_test=False, add_avg_over_datasets=True):
    # read df
    df = pd.read_csv(input_file)

    if add_avg_over_datasets:
        df = add_all_cols_group(df, 'dataset_name')

    splits = ['train']
    if include_test:
        splits.append('test')

    for split in splits:

        # cka
        cols_cka_split = get_cols(df.columns, 'cka', split)

        df[f'cka_mean_{split}'] = df[cols_cka_split].mean(axis=1)
        df[f'cka_std_{split}'] = df[cols_cka_split].std(axis=1)

        # cka_change_mean
        cols_cka_change_mean_split = get_cols(df.columns, 'cka_change_mean', split)

        df[f'cka_change_mean_mean_{split}'] = df[cols_cka_change_mean_split].mean(axis=1)
        df[f'cka_change_mean_std_{split}'] = df[cols_cka_change_mean_split].std(axis=1)

        # cka_change2
        cols_cka_change2_mean_split = get_cols(df.columns, 'cka_change2_mean', split)

        df[f'cka_change2_mean_mean_{split}'] = df[cols_cka_change2_mean_split].mean(axis=1)
        df[f'cka_change2_mean_std_{split}'] = df[cols_cka_change2_mean_split].std(axis=1)

        # dist
        cols_dist_split = get_cols(df.columns, 'dist', split)

        df[f'dist_mean_{split}'] = df[cols_dist_split].mean(axis=1)
        df[f'dist_std_{split}'] = df[cols_dist_split].std(axis=1)

        # dist_norm
        cols_dist_norm_split = get_cols(df.columns, 'dist_norm', split)

        df[f'dist_norm_mean_{split}'] = df[cols_dist_norm_split].mean(axis=1)
        df[f'dist_norm_std_{split}'] = df[cols_dist_norm_split].std(axis=1)

        # l2_norm
        cols_l2_norm_split = get_cols(df.columns, 'l2_norm', split)

        df[f'l2_norm_mean_{split}'] = df[cols_l2_norm_split].mean(axis=1)
        df[f'l2_norm_std_{split}'] = df[cols_l2_norm_split].std(axis=1)

        # attn_mean
        cols_attn_mean_split = get_cols(df.columns, 'attn_mean', split)

        df[f'attn_mean_mean_{split}'] = df[cols_attn_mean_split].mean(axis=1)
        df[f'attn_mean_std_{split}'] = df[cols_attn_mean_split].std(axis=1)
        df[f'attn_mean_last_first_ratio_{split}'] = df[f'attn_mean_11_{split}'] / df[f'attn_mean_0_{split}'] 

        # attn_std
        cols_attn_std_split = get_cols(df.columns, 'attn_mean', split)

        df[f'attn_std_mean_{split}'] = df[cols_attn_std_split].mean(axis=1)
        df[f'attn_std_std_{split}'] = df[cols_attn_std_split].std(axis=1)


    keep_cols = [c for c in df.columns if any([kw in c for kw in ['mean_train', 'mean_test', 'std_train', 'std_test']])]
    id_cols = ['serial', 'dataset_name', 'setting', 'compute_attention_average', 'compute_attention_cka']
    # cols = id_cols + keep_cols

    df = df.groupby(id_cols, as_index=False).agg({c: 'mean' for c in keep_cols})

    # df = df[cols].round(3)
    df = df.round(3)
    df.to_csv(output_file, header=True, index=False)

    return df


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('data', 'saw_feature_metrics.csv'),
                        help='filename for input .csv file from wandb')
    parser.add_argument('--include_test', action='store_true')
    # output
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'feature_metrics'),
                        help='The directory where results will be stored')
    parser.add_argument('--output_file', type=str, default='feature_metrics_mean_std.csv',
                        help='filename for output .csv file')
    args= parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    df = process_df(args.input_file, args.output_file, args.include_test)

    return df



if __name__ == '__main__':
    main()