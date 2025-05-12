import os
import argparse

import pandas as pd

from utils import preprocess_df, sort_df, group_by_family, \
    DATASETS_UFGIR, METHODS_DIC


def summarize_test_cost(input_file, host='server-3090', keep_serials=[40, 41]):
    df = preprocess_df(
        input_file,
        'inference_cost',
        keep_serials=keep_serials,
    )


    if 'DGX' in host:
        df = df[(df['host'].str.contains(host))].copy(deep=False)
    else:
        df = df[(df['host'] == host)].copy(deep=False)


    best_list = []

    method_list = df['method'].unique()
    for method in method_list:
        for setting in df[(df['method'] == method)]['setting'].unique():
            df_subset = df[
                (df['method'] == method) & (df['setting'] == setting)
            ]

            df_subset = df_subset.dropna(subset=['throughput'])
            df_subset_sorted = df_subset.sort_values(by=['throughput'], ascending=False)

            bs_batched = df_subset_sorted['batch_size'].iloc[0]
            tp_batched = df_subset_sorted['throughput'].iloc[0]
            vram_batched = df_subset_sorted['max_memory'].iloc[0]

            df_subset_sorted = df_subset.sort_values(by=['batch_size'], ascending=True)

            tp_stream = df_subset_sorted['throughput'].iloc[0]
            latency_stream = 1 / tp_stream
            vram_stream = df_subset_sorted['max_memory'].iloc[0]

            # flops = df_subset_sorted['flops'].iloc[0]

            serial = df_subset['serial'].iloc[0]

            method_rename = f'{method}_fz'

            # save tp and vram for each method-setting pair
            best_list.append({
                'serial': serial, 'method': method_rename, # 'flops_inference': flops,
                'tp_stream': tp_stream, 'vram_stream': vram_stream, 'latency_stream': latency_stream,
                'tp_batched': tp_batched, 'vram_batched': vram_batched, 'bs_batched': bs_batched
            })


    df = pd.DataFrame.from_dict(best_list)


    # saw uses same inference resources as ila
    saw = df[df['method'] == 'vit_b16_ila_dso_cls_adapter_fz'].copy(deep=False)
    saw['method'] = saw['method'].str.replace('_fz', '_saw_fz')

    df = pd.concat([df, saw], axis=0, ignore_index=True)

    return df


def agg_train_time_tp_vram(df):
    # convert train time to hours, compute tp_train (imgs/s), rename memory
    df['time_train'] = df['time_total'] / 60
    df['tp_train'] = ((df['num_images_val'] + (df['num_images_train'] * df['epochs'])) 
        / (df['time_total'] * 60))
    df['vram_train'] = df['max_memory']


    # aggregate across seeds
    df = df.groupby(['serial', 'method'], as_index=False).agg({
        'time_train': 'mean',
        'tp_train': 'mean',
        'vram_train': 'mean',
    })

    return df


def summarize_train_cost(input_file, keep_serials=[42, 43]):
    df = preprocess_df(
        input_file,
        'train_cost',
        keep_serials=keep_serials,
    )


    # aggregate train cost results
    df = agg_train_time_tp_vram(df)


    # keep relevant columns
    cols_keep = ['serial', 'method', 'time_train', 'tp_train', 'vram_train']
    df = df[cols_keep]


    # saw uses same train resources during ft as ila
    saw = df[df['method'] == 'vit_b16_ila_dso_cls_adapter_fz'].copy(deep=False)
    saw['method'] = saw['method'].str.replace('_fz', '_saw_fz')

    df = pd.concat([df, saw], axis=0, ignore_index=True)

    return df


def agg_train_flops(df, ds='soylocal', modify_serials=[90, 100, 210, 220, 250]):
    if ds:
        # modify leaves to ds otherwise params empty
        df.loc[df["serial"].isin(modify_serials), "dataset_name"] = ds

        df = df[(df['dataset_name'] == ds)].copy(deep=False)

    df = df.groupby(['serial', 'method'], as_index=False).agg({
        'flops': 'mean',
    })
    # df = df.rename(columns={'flops': 'flops_train'})
    return df


def summarize_train_flops(
    input_file,
    keep_serials=[1, 3, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 71, 81, 90, 91, 100, 101, 210, 220, 250]):
    df = preprocess_df(
        input_file,
        'train_cost',
        keep_serials=keep_serials,
    )

    # aggregate parameters and flops and combine
    df = agg_train_flops(df)

    # keep relevant columns
    cols_keep = ['serial', 'method', 'flops',]
    df = df[cols_keep]


    return df


def agg_parameters(df, ds='soylocal', modify_serials=[90, 100, 210, 220, 250]):
    df = df.copy(deep=False)

    # aggregate across seeds
    df = df.groupby(['serial', 'method', 'dataset_name'], as_index=False).agg({
       'no_params': 'mean',
       'no_params_trainable': 'mean',
    })


    # mean aggregate across datasets
    # df_agg = df.groupby(['serial', 'method'], as_index=False).agg({
    #     'no_params': 'mean',
    #     'no_params_trainable': 'mean',
    # })

    # aggregate for one single dataset if possible
    if ds:
        # modify leaves to ds otherwise params empty
        df.loc[df["serial"].isin(modify_serials), "dataset_name"] = ds

        df_agg = df[(df['dataset_name'] == ds)].copy(deep=False)
        df_agg = df_agg.groupby(['serial', 'method'], as_index=False).agg({
            'no_params': 'mean',
            'no_params_trainable': 'mean',
        })

    else:
        df_agg = df.groupby(['serial', 'method'], as_index=False).agg({
            'no_params': 'mean',
            'no_params_trainable': 'mean',
        })

    # sum aggregate across datasets
    df_agg_sum = df.groupby(['serial', 'method'], as_index=False).agg({
        'no_params': 'sum',
        'no_params_trainable': 'sum',
    })

    renames = {
        'no_params': 'no_params_total',
        'no_params_trainable': 'no_params_trainable_total'
    }
    df_agg_sum = df_agg_sum.rename(columns=renames)


    # combine mean and total aggregates
    df = pd.merge(df_agg, df_agg_sum,
                  how='left', on=['serial', 'method'])


    # percentage of trainable parameters
    df['trainable_percent'] = 100 * (df['no_params_trainable'] / df['no_params'])

    return df


def summarize_parameters(input_file, keep_datasets=None, keep_methods=None,
    keep_serials=[1, 3, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 71, 81, 90, 91, 100, 101, 210, 220, 250]):
    df = preprocess_df(
        input_file,
        'train_cost',
        keep_datasets,
        keep_methods,
        keep_serials,
    )

    # aggregate parameters
    df = agg_parameters(df)

    # keep relevant columns
    cols_keep = ['serial', 'method',
                 'trainable_percent', 'no_params', 'no_params_trainable',
                 'no_params_total', 'no_params_trainable_total']
    df = df[cols_keep]


    return df


def summarize_acc(
    input_file, keep_datasets=None, keep_methods=None,
    keep_serials=[1, 3, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 71, 81, 90, 91, 100, 101, 210, 220, 250]):
    df = preprocess_df(
        input_file,
        'acc',
        keep_datasets,
        keep_methods,
        keep_serials,
    )


    df_std = df.groupby(['serial', 'setting','method'], as_index=False).agg({'acc': 'std'})
    df = df.groupby(['serial', 'setting', 'method'], as_index=False).agg({'acc': 'mean'})
    df['acc_std'] = df_std['acc']

    return df


def summarize_acc_cost(args):
    # read acc, train_cost and test_cost
    df_acc = summarize_acc(
        args.input_file_acc,
        getattr(args, 'keep_datasets', None), getattr(args, 'keep_methods', None))

    df_parameters = summarize_parameters(args.input_file_acc,
        getattr(args, 'keep_datasets', None), getattr(args, 'keep_methods', None))

    df_train_flops = summarize_train_flops(args.input_file_acc)

    df_train_cost = summarize_train_cost(args.input_file_train_cost)

    df_test_cost = summarize_test_cost(args.input_file_inference_cost, args.host)


    # combine acc, train and test cost and sort based on method
    # outer if want to keep even if not all match
    df = pd.merge(df_acc, df_parameters, how='left', on=['serial', 'method'])
    df = pd.merge(df, df_train_flops, how='left', on=['serial', 'method'])
    df = pd.merge(df, df_train_cost, how='left', on=['serial', 'method'])
    df = pd.merge(df, df_test_cost, how='left', on=['serial', 'method'])
    df = sort_df(df, method_only=True)


    # add column that groups up into Classifiers, PEFGIR, PETL, Ours
    df['family'] = df['method'].apply(group_by_family)


    # aggregate and save results
    fp = os.path.join(args.results_dir, f'{args.output_file}.csv')
    df.to_csv(fp, header=True, index=False)

    return df



def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file_acc', type=str, 
                        default=os.path.join('data', 'saw_stage2.csv'),
                        help='filename for input .csv file from wandb')
    parser.add_argument('--input_file_train_cost', type=str,
                        default=os.path.join('data', 'saw_cost.csv'),
                        help='filename for input .csv file from wandb')
    parser.add_argument('--input_file_inference_cost', type=str, 
                        default=os.path.join('data', 'saw_cost.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--host', type=str, default='server-3090')

    parser.add_argument('--keep_datasets', nargs='+', type=str,
                        default=DATASETS_UFGIR)
    parser.add_argument('--keep_methods', nargs='+', type=str,
                        default=METHODS_DIC.keys())

    # output
    parser.add_argument('--output_file', type=str, default='cost',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'cost'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df = summarize_acc_cost(args)
    return df


if __name__ == '__main__':
    main()
