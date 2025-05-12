import os
import argparse
import pandas as pd


def get_best_lr_opt_wd(
    df_subset, dataset, method,
    selection_var='val_acc', lr_var='lr', train_acc_th=50
    ):

    # sort subset based on val loss (lowest to highest)
    ascending = True if 'loss' in selection_var else False
    df_subset_sorted = df_subset.sort_values(by=[selection_var], ascending=ascending)

    # get index and train acc corresponding to best val loss (0)
    best_idx = 0
    train_acc = df_subset_sorted['train_acc'].iloc[best_idx]

    while train_acc < train_acc_th and best_idx < (len(df_subset) - 1):
        # increment index of next best val loss
        best_idx += 1
        # get val loss and train acc corresponding to next best
        train_acc = df_subset_sorted['train_acc'].iloc[best_idx]
        # print when train_acc < 50 (mostly cotton/soy/ultra fine-grained datasets)
        print(f'Acc below {train_acc_th}: {dataset}, {method}, {best_idx}, {train_acc}')
    
    # if none of the accuracies surpass TH then just return the highest accuracies
    if train_acc < train_acc_th:
        # sort subset based on train acc (highest to lowest)
        df_subset_sorted = df_subset.sort_values(by=['train_acc'], ascending=False) 
        print(f'All below train acc threshold ({train_acc_th}): {dataset}, {method}')
        print(df_subset_sorted)
        lr = df_subset_sorted[lr_var].iloc[0]
        opt = df_subset_sorted['opt'].iloc[0]
        wd = df_subset_sorted['weight_decay'].iloc[0]

    # returns lr corresponding to best val loss
    else:
        lr = df_subset_sorted[lr_var].iloc[best_idx]
        opt = df_subset_sorted['opt'].iloc[best_idx]
        wd = df_subset_sorted['weight_decay'].iloc[best_idx]
    
    # check the current subset and the selected LR
    # print(dataset, method, lr, best_idx, df_subset.head())

    return lr, opt, wd


def get_lr_cmd(
    df_subset, dataset, method, suffix, prefix,
    selection_var='val_acc', lr_var='lr', train_acc_th=50
    ):
    lr, opt, wd = get_best_lr_opt_wd(
        df_subset, dataset, method, selection_var, lr_var, train_acc_th
    )

    #write to file
    model_name = df_subset['model_name'].iloc[0]
    fz = ' --freeze_backbone' if 'fz' in method else ''

    classifier = df_subset['classifier'].iloc[0]
    classifier = f" --classifier {classifier.replace('_', '', 1)}" if classifier else ''

    adapter = df_subset['adapter'].iloc[0]
    adapter = f" --adapter {adapter.replace('_', '', 1)}" if adapter else ''

    prompt = df_subset['prompt'].iloc[0].replace('_', '', 1)
    prompt = f" --prompt {prompt}" if prompt else ''

    ila = df_subset['ila'].iloc[0]
    ila_arch = df_subset['ila_arch'].iloc[0]
    if adapter and ila:
        ila = f' --ila --ila_arch {ila_arch} --ila_locs'
    elif ila:
        ila = f' --ila --ila_arch {ila_arch}'
    else:
        ila = ''

    opt_text = f' --opt {opt} --weight_decay {wd}'
    method_text = f'{fz}{classifier}{adapter}{prompt}{ila}{opt_text}'
    others = f'{method_text}{suffix}'

    line = f'{prefix} --cfg configs/{dataset}_ft_weakaugs.yaml --{lr_var} {lr} --model_name {model_name}{others}\n'
    return line


def make_lr_script(args):
    df = pd.read_csv(args.input_file)
    df = df[['dataset_name', 'model_name', 'freeze_backbone',
             'classifier', 'adapter', 'prompt',
             'ila', 'ila_arch',
             args.selection_var, 'train_acc', 'lr', 'base_lr', 'opt', 'weight_decay']]

    # dataset and method names
    dataset_list = df['dataset_name'].unique()

    df = df.fillna({'classifier': '', 'adapter': '', 'prompt': ''})

    df['freeze_backbone'] = df['freeze_backbone'].apply(lambda x: '_fz' if x is True else '')
    df['classifier'] = df['classifier'].apply(lambda x: f'_{x}' if x else '')
    df['adapter'] = df['adapter'].apply(lambda x: f'_{x}' if x else '')
    df['prompt'] = df['prompt'].apply(lambda x: f'_{x}' if x else '')
    df['ila'] = df['ila'].apply(lambda x: f'_ila' if x else '')

    df['method'] = df['model_name'] + df['classifier'] + df['adapter'] + df['prompt'] + df['freeze_backbone']

    # create file with specified file name
    output_file = os.path.join(args.results_dir, f'{args.output_file}.sh')
    f = open(output_file, "w")

    #for loop for each dataset
    for dataset in dataset_list:
        # write dataset name
        f.write(f'# {dataset}\n')

        prefix = args.prefix
        lr_var = args.lr_var
        suffix = args.suffix

        # for loop for each method
        method_list = df[df['dataset_name'] == dataset]['method'].unique()

        for method in method_list:
            # filter the subset of the dataset based on the dataset and the method
            df_subset = df[(df['method'] == method) & (df['dataset_name'] == dataset)].copy(deep=False)
            df_subset.dropna(subset=args.selection_var, inplace=True)

            if len(df_subset) == 0:
                print(dataset, method)
                continue
            
            line = get_lr_cmd(
                df_subset, dataset, method, suffix, prefix,
                args.selection_var, lr_var, args.train_acc_th
            )

            f.write(line)
        f.write('\n')

    f.close()

    return 0


def parse_args():

    parser = argparse.ArgumentParser()

    #parser arguments
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('data', 'aaa_stage1.csv'),
                        help='filename for input .csv file')

    parser.add_argument('--selection_var', type=str, default='val_acc')
    parser.add_argument('--lr_var', type=str, default='lr',
                        help='for inat/dafb use --lr_var base_lr')
    parser.add_argument('--train_acc_th', type=int, default=50)
    parser.add_argument('--prefix', type=str,
                        default='python -u tools/train.py --serial 1 --seed 1',
                        help='prefix for the file in each line')
    parser.add_argument('--suffix', type=str,
                        default=' --cpu_workers 8',
                        help='suffix for the file in each line')
    parser.add_argument('--output_file', type=str, default='aaa_stage2',
                        help='output file name')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'lr_scripts'),
                        help='The directory where results will be stored')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    make_lr_script(args)

    return 0


if __name__ == '__main__':
    main()

