import os
import argparse

import pandas as pd

from utils import preprocess_df, add_all_cols_group, \
    round_combine_str_mean_std, sort_df, rename_var, \
    DATASETS_UFGIR, DATASETS_DIC, METHODS_DIC


def aggregate_results_main(df, fp=None, serials=[1, 3],
                           add_method_avg=True, add_dataset_avg=False):
    df = df[df['serial'].isin(serials)]


    if add_method_avg:
        df = add_all_cols_group(df, 'dataset_name')
    if add_dataset_avg:
        df = add_all_cols_group(df, 'method')


    df_std = df.groupby(['serial', 'setting', 'dataset_name', 'method'], as_index=False).agg({'acc': 'std'})
    df = df.groupby(['serial', 'setting', 'dataset_name', 'method'], as_index=False).agg({'acc': 'mean'})
    df['acc_std'] = df_std['acc']


    df = sort_df(df)


    df = round_combine_str_mean_std(df, col='acc')


    if fp:
        df.to_csv(fp, header=True, index=False)
    return df


def aggregate_results_ablations(df, fp=None):
    df_bl = df[(df['serial'].isin([1]) & (df['method'].str.contains('ila'))) &
               (df['dataset_name'].isin(['soyglobal', 'soylocal']))]
    df_bl_saw = df[(df['serial'].isin([61, 71, 81, 91, 211, 221, 231, 241, 251, 261])
                    & (df['method'].str.contains('ila'))) &
               (df['dataset_name'].isin(['soyglobal', 'soylocal']))]
    df = df[df['serial'].isin([4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18,
                               20, 21, 22, 23, 24, 25, 27])]

    df = pd.concat([df_bl, df_bl_saw, df], ignore_index=True)


    df_std = df.groupby(['serial', 'setting', 'dataset_name', 'method'], as_index=False).agg({'acc': 'std'})
    df = df.groupby(['serial', 'setting', 'dataset_name', 'method'], as_index=False).agg({'acc': 'mean'})
    df['acc_std'] = df_std['acc']


    df = sort_df(df)


    df = round_combine_str_mean_std(df, col='acc')


    # to do:
    # compute the absolute differences to all of this similar to pivot_table compute_diffs


    if fp:
        df.to_csv(fp, header=True, index=False)

    return df


def compute_diffs(df):
    ours_list = [m for m in df.index if 'ila' in m]
    methods = [m for m in df.index if 'ila' not in m]

    for ours in ours_list:
        for i, (index, row) in enumerate(df.iterrows()):
            if index != ours and index in methods:
                df.loc[f'diff_{ours}_{str(index)}'] = df.loc[df.index == ours].iloc[0] - row 

    return df


def pivot_table(df, serial=1, fp=None, var='acc', save_diff=False, rename=True):
    df = df[df['serial'] == serial].copy(deep=False)


    df = df.pivot(index='method', columns='dataset_name')[var]


    datasets = [ds for ds in DATASETS_DIC.keys() if ds in df.columns]
    df = df[datasets]


    df['method_order'] = pd.Categorical(df.index, categories=METHODS_DIC.keys(), ordered=True)
    df = df.sort_values(by=['method_order'], ascending=True)
    df = df.drop(columns=['method_order'])


    if save_diff:
        df = compute_diffs(df)


    if rename:
        df.index = df.index.map(rename_var)
        df.columns = df.columns.map(rename_var)


    if fp:
        df.to_csv(fp, header=True, index=True)
    else:
        print(df)

    return 0


def summarize_acc(args):
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

    df_main = aggregate_results_main(df, f'{fp}_main.csv', args.main_serials)


    for serial in args.main_serials:
        df_pivoted = pivot_table(df_main, serial, f'{fp}_{serial}_pivoted.csv',
                                 var='acc', save_diff=True)
        pivot_table(df_main, serial, f'{fp}_{serial}_pivoted_mean_std.csv',
                                 var='acc_mean_std')
        pivot_table(df_main, serial, f'{fp}_{serial}_pivoted_mean_std_latex.csv',
                                 var='acc_mean_std_latex')


    # aggregate and save ablations results
    df_ablations = aggregate_results_ablations(df, f'{fp}_ablations.csv')

    return df, df_main, df_ablations


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('data', 'saw_stage2.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=DATASETS_UFGIR)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=METHODS_DIC.keys())
    parser.add_argument('--main_serials', nargs='+', type=int, default=[1, 3])

    # output
    parser.add_argument('--output_file', type=str, default='acc',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'acc'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df = summarize_acc(args)
    return df


if __name__ == '__main__':
    main()
