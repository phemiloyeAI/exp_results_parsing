import os
import argparse
import wandb
import pandas as pd


CONFIG_COLS = [
    'serial', 'dataset_name', 'model_name', 'freeze_backbone', 'transfer_learning',

    'classifier', 'selector', 'prompt', 'adapter',
    'ila', 'ila_padding', 'ila_ds_conv', 'ila_ds_conv_type',
    'ila_locs', 'ila_ds_locs', 'ila_ds_kernel_size', 'ila_arch',

    'lr', 'base_lr', 'opt', 'weight_decay', 'seed', 'epochs', 'image_size',
    'batch_size', 'num_images_train', 'num_images_val',
]

SUMMARY_COLS = [
    'val_acc', 'val_loss', 'train_acc', 'train_loss',
    'time_total', 'max_memory', 'flops',
    'no_params', 'no_params_trainable', 'throughput',
]

SORT_COLS = [
    'dataset_name', 'serial', 'model_name', 'ila',
    'lr', 'seed', 'host', 'batch_size',
]


def get_wandb_project_runs(project, serials=None):
    api = wandb.Api()

    if serials:
        # runs = api.runs(path=project, per_page=2000,
        #                filters={'$or': [{'config.serial': s} for s in serials]})
        runs = api.runs(path=project, per_page=2000,
                        # filters={'$and': [{'config.serial': 1}, {'config.dataset_name': 'cub'}]}
                        # filters={'$or': [{'config.serial': s} for s in serials]}
                        # nin (not in) also an option
                        filters={'config.serial': {'$in': serials}}
                        )
    else:
        runs = api.runs(path=project, per_page=2000)

    print('Downloaded runs: ', len(runs))
    return runs


def make_df(runs, config_cols, summary_cols):
    data_list_dics = []

    for i, run in enumerate(runs):
        run_data = {}
        try:
            host = {'host': run.metadata.get('host')}
        except:
            print(run)
            host = {'host': None}
        cfg = {col: run.config.get(col, None) for col in config_cols}
        summary = {col: run.summary.get(col, None) for col in summary_cols}

        run_data.update(host)
        run_data.update(cfg)
        run_data.update(summary)

        data_list_dics.append(run_data)

        if (i + 1) % 100 == 0:
            print(f'{i}/{len(runs)}')

    df = pd.DataFrame.from_dict(data_list_dics)
    print(df.head())

    return df


def sort_save_df(df, fp, sort_cols=['serial']):
    df = df.sort_values(by=sort_cols, ascending=[True for _ in sort_cols])
    df.to_csv(fp, header=True, index=False)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--project_name', type=str, default='nycu_pcs/ILA',
                        help='project_entity/project_name')
    # filters
    parser.add_argument('--serials', nargs='+', type=int,
                        default=[1, 3, 4, 5, 6, 7, 8, 9,
                                 10, 11, 12, 14, 15, 16, 17, 18,
                                 20, 21, 22, 23, 24, 25, 27,
                                 61, 71, 81, 90, 91, 92, 93, 100, 101, 102, 103,
                                 210, 211, 220, 221, 230, 231, 240, 241,
                                 250, 251, 260, 261])
    parser.add_argument('--config_cols', nargs='+', type=str, default=CONFIG_COLS)
    parser.add_argument('--summary_cols', nargs='+', type=str, default=SUMMARY_COLS)
    # output
    parser.add_argument('--output_file', default='saw_stage2.csv', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str, default='data',
                        help='The directory where results will be stored')
    parser.add_argument('--sort_cols', nargs='+', type=str, default=SORT_COLS)

    args= parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    runs = get_wandb_project_runs(args.project_name, args.serials)

    df = make_df(runs, args.config_cols, args.summary_cols)

    sort_save_df(df, args.output_file, args.sort_cols)

    return 0

if __name__ == '__main__':
    main()
