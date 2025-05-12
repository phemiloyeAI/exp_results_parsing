import os
import argparse
from numbers import Number
import wandb
import pandas as pd

CONFIG_COLS = [
    'serial', 'dataset_name', 'model_name', 'freeze_backbone',
    'compute_attention_average', 'compute_attention_cka', 'setting',

    'classifier', 'selector', 'prompt', 'adapter',
    'ila', 'transfer_learning', 'ssl',
]

SORT_COLS = [
    'dataset_name', 'compute_attention_average', 'compute_attention_cka',
    'serial', 'setting', 'model_name',
]


def get_wandb_project_runs(project, serials=None):
    api = wandb.Api()

    summary_cols = []

    if serials:
        for s in serials:
            if s == 31:
                runs = api.runs(path=project, per_page=2000,
                                filters={'$and': [{'config.serial': s},
                                                  {'config.compute_attention_average': True}]})

                run = runs[0]
                cols = [c for c in run.summary.keys() if 'attn_' in c]
                cols = [col for col in cols if isinstance(run.summary.get(col, 0), Number)]
                summary_cols.extend(cols)

            elif s == 32:
                runs = api.runs(path=project, per_page=2000,
                                filters={'$or': [{'config.serial': s}]})

                run = runs[0]
                cols = [c for c in run.summary.keys() if any(sub in c for sub in ['cka_', 'dist_', 'l2_'])]
                cols = [col for col in cols if isinstance(run.summary.get(col, 0), Number)]
                summary_cols.extend(cols)

        runs = api.runs(path=project, per_page=2000,
                        filters={'$or': [{'config.serial': s} for s in serials]})

    else:
        raise NotImplementedError

    print('Downloaded runs: ', len(runs))
    return runs, summary_cols


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
                        default=[31, 32])
    parser.add_argument('--config_cols', nargs='+', type=str, default=CONFIG_COLS)
    # output
    parser.add_argument('--output_file', default='saw_feature_metrics.csv', type=str,
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

    runs, summary_cols = get_wandb_project_runs(args.project_name, args.serials)

    df = make_df(runs, args.config_cols, summary_cols)

    sort_save_df(df, args.output_file, args.sort_cols)

    return 0

if __name__ == '__main__':
    main()
