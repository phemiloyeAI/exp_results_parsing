import numpy as np
import pandas as pd


SERIALS_EXPLANATIONS = [
    # main experiments
    'fz_224',
    'fz_448',

    # ablation on compatibility with other petl methods
    # combine with convpass, ila (all locs) + adapter,
    # ila (ds only) + adapters + vpt shallow +
    # ila (all locs) + adapters + vpt shallow
    'compatibility_with_other_petl',

    # ablation on downsampling conv (no downsampling, use padding to use skip connection, increase flops)
    'no_downsampling',
    # ablation on downsampling with no padding (no skip connection, importance of skip connection)
    'downsampling_no_padding_no_skip',

    # rsds design (residual downsampling design: near ones init vs dws, conv, pool/avg)
    'rsds_design_dws',
    'rsds_design_conv',
    'rsds_design_avg_pool',

    # ila downsampling kernel size
    'ila_ds_kernel_size_1',
    'ila_ds_kernel_size_5',
    'ila_ds_kernel_size_7',

    # ila downsampling location
    'ila_ds_loc_0_1',
    'ila_ds_loc_1_3',
    'ila_ds_loc_2_5',
    'ila_ds_loc_4_9',
    'ila_ds_loc_5_11',

    # ila_cls_dw cls token adapter dw convolution
    'ila_cls_dw',

    # ila_arch
    'ila_pool',
    'ila_pool_conv',
    'ila_pool_attn',
    'ila_attn',
    'ila_attn_agg',

    # different pretrain strategies/backbones
    'fz_224',

    # inference cost
    'inference_224',
    'inference_448',

    # train cost
    'train_224',
    'train_448',

    # saw
    'saw_rollout_0_4',

    # different pretrain strategies/backbones
    'saw_224',
    'saw_448',

    'saw_cost_224',
    'saw_224',
    'saw_mix_224',
    'saw_mix_cl_224',

    'saw_cost_448',
    'saw_448',
    'saw_mix_448',
    'saw_mix_cl_448',

    'saw_cost_pid',
    'saw_pid',
    'saw_cost_selfcon',
    'saw_selfcon',

    'saw_cost_crop',
    'saw_crop',
    'saw_cost_mask',
    'saw_mask',

    'saw_cost_single_dataset',
    'saw_single_dataset',

    'saw_cost_generic_aug',
    'saw_generic_aug',
]


SETTINGS_DIC = {
    'fz_224': 'IS=224',
    'fz_448': 'IS=448',
    'ft_224': 'IS=224 (FT)',
    'ft_448': 'IS=448 (FT)',
    'saw_224': 'IS=224 (SAW)',
    'saw_448': 'IS=448 (SAW)',
}


DATASETS_UFGIR = [
    'cotton',
    'soyageing',
    'soyageingr1',
    'soyageingr3',
    'soyageingr4',
    'soyageingr5',
    'soyageingr6',
    'soygene',
    'soyglobal',
    'soylocal',
    'leaves',
]


DATASETS_DIC = {
    'cotton': 'Cotton',
    'cub': 'CUB',
    'soyageing': 'SoyAgeing',
    'soyageingr1': 'SoyAgeR1',
    'soyageingr3': 'SoyAgeR3',
    'soyageingr4': 'SoyAgeR4',
    'soyageingr5': 'SoyAgeR5',
    'soyageingr6': 'SoyAgeR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'leaves': 'Leaves',
    'all': 'Average',
}


METHODS_DIC = {
    # method families
    'classifier': 'Classifier',
    'fgir': 'FT FGIR',
    'ufgir': 'FT FGIR',
    'petl': 'PETL',
    'pefgir': 'PEFGIR',
    'peclassifier': 'PE Classifier',
    'ila': 'ILA (Ours)',
    'saw': 'SAW (Ours)',


    # classifiers (technically pefgir: fgir methods in parameter-efficient setting)
    'vit_b16_cls': 'Baseline', 
    'vit_b16_lrblp': 'LR-BLP',
    'vit_b16_mpncov': 'MPN-COV',
    'vit_b16_ifacls': 'IFA',

    # fgir methods
    'vit_b16_cls_psm': 'TransFG',
    'vit_b16_cls_maws': 'FFVT',
    'vit_b16_cal': 'CAL',
    'vit_b16_avg_cls_rollout': 'RAMS-Trans',
    'vit_b16_cls_glsim': 'GLSim',

    # ufgir methods
    'vit_b16_sil': 'SimTrans',
    'mixvit': 'MixViT',
    'clevit': 'CLE-ViT',
    'csdnet': 'CSDNet',

    'vit_b16_sil_fz': 'SimTrans',
    'mixvit_fz': 'MixViT',
    'clevit_fz': 'CLE-ViT',
    'csdnet_fz': 'CSDNet',


    # pe classifiers (technically pefgir: fgir methods in parameter-efficient setting)
    'vit_b16_cls_fz': 'Baseline',
    'vit_b16_lrblp_fz': 'LR-BLP',
    'vit_b16_mpncov_fz': 'MPN-COV',
    'vit_b16_ifacls_fz': 'IFA',

    # fgir methods in parameter-efficient setting: pefgir
    'vit_b16_cls_psm_fz': 'TransFG',
    'vit_b16_cls_maws_fz': 'FFVT',
    'vit_b16_cal_fz': 'CAL',
    'vit_b16_avg_cls_rollout_fz': 'RAMS-Trans',
    'vit_b16_cls_glsim_fz': 'GLSim',

    # petl methods
    'vit_b16_cls_vqt_fz': 'VQT',
    'vit_b16_cls_vpt_shallow_fz': 'VPT-Shallow',
    'vit_b16_cls_vpt_deep_fz': 'VPT-Deep',
    'vit_b16_cls_convpass_fz': 'ConvPass',
    'vit_b16_cls_adapter_fz': 'Adapter',


    # 3 main variations of our method (in serial 1 and 3)
    'vit_b16_ila_dso_cls_fz': 'ILA',
    'vit_b16_ila_cls_fz': 'ILA+',
    'vit_b16_ila_dso_cls_adapter_fz': 'ILA++',
    # extra variations (serial 4 ablation, combined with convpass or vpt_shallow)
    'vit_b16_ila_dso_cls_convpass_fz': 'ILA++ (ConvPass)',
    'vit_b16_ila_cls_adapter_fz': 'ILA+++',
    'vit_b16_ila_dso_cls_adapter_vpt_shallow_fz': 'ILA++ (VPT)',
    'vit_b16_ila_cls_adapter_vpt_shallow_fz': 'ILA+++ (VPT)',

    # saw
    'vit_b16_ila_dso_cls_adapter_saw_fz': 'DAMAS',

    # different pt strategy / backbone
    'pedeit_base_patch16_224.fb_in1k_cls_fz': 'Baseline (DeiT)',
    'pedeit3_base_patch16_224.fb_in1k_cls_fz': 'Baseline (DeiT3)',
    'pedeit_base_patch16_224.fb_in1k_cls_adapter_fz': 'Adapter (DeiT)',
    'pedeit3_base_patch16_224.fb_in1k_cls_adapter_fz': 'Adapter (DeiT3)',
    'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz': 'ILA++ (DeiT)',
    'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz': 'ILA++ (DeiT3)',
    'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz': 'DAMAS (DeiT)',
    'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz': 'DAMAS (DeiT3)',

}


VAR_DIC = {
    'setting': 'Setting',
    'acc': 'Accuracy (%)',
    'acc_std': 'Accuracy Std. Dev. (%)',
    'dataset_name': 'Dataset',
    'method': 'Method',
    'family': 'Method Family',
    'flops': 'Inference FLOPs (10^9)',
    'time_train': 'Train Time (hours)',
    'vram_train': 'Train VRAM (GB)',
    'tp_train': 'Train Throughput (Images/s)',
    'trainable_percent': 'Task-Specific Trainable Parameters (%)',
    'no_params': 'Number of Parameters (10^6)',
    'no_params_trainable': 'Task-Specific Trainable Parameters (10^6)',
    'no_params_total': 'Total Parameters (10^6)',
    'no_params_trainable_total': 'Total Task-Specific Trainable Params. (10^6)',
    'flops_inference': 'Inference FLOPs (10^9)',
    'tp_stream': 'Stream Throughput (Images/s)',
    'vram_stream': 'Stream VRAM (GB)',
    'latency_stream': 'Stream Latency (s)',
    'tp_batched': 'Batched Throughput  (Images/s)',
    'vram_batched': 'Batched VRAM (GB)',
}


SERIAL_REASSIGN = {
    40: 1,
    41: 3,
    42: 1,
    43: 3,

    71: 1,
    81: 3,

    91: 1,
    # 92: 1,
    # 93: 1,
    101: 3,
    # 102: 3,
    # 103: 3,
}


def rename_serial(x):
    return SERIAL_REASSIGN.get(x, x)


def reassign_serial(df):
    df['serial'] = df['serial'].apply(rename_serial)
    return df


def rename_var(x):
    if x in SETTINGS_DIC.keys():
        return SETTINGS_DIC[x]
    elif x in METHODS_DIC.keys():
        return METHODS_DIC[x]
    elif x in DATASETS_DIC.keys():
        return DATASETS_DIC[x]
    elif x in VAR_DIC.keys():
        return VAR_DIC[x]
    return x


def rename_vars(df, var_rename=False, args=None):
    if 'setting' in df.columns:
        df['setting'] = df['setting'].apply(rename_var)
    if 'method' in df.columns:
        df['method'] = df['method'].apply(rename_var)
    if 'dataset_name' in df.columns:
        df['dataset_name'] = df['dataset_name'].apply(rename_var)
    if 'family' in df.columns:
        df['family'] = df['family'].apply(rename_var)

    if var_rename:
        df.rename(columns=VAR_DIC, inplace=True)
        for k, v in VAR_DIC.items():
            if k == args.x_var_name:
                args.x_var_name = v
            elif k == args.y_var_name:
                args.y_var_name = v
            elif k == args.hue_var_name:
                args.hue_var_name = v
            elif k == args.style_var_name:
                args.style_var_name = v
            elif k == args.size_var_name:
                args.size_var_name = v

    return df


def determine_ila_method(row):
    if row['ila'] == True and row['ila_locs'] == '[]':
        return "_ila_dso"
    elif row['ila'] == True:
        return "_ila"
    else:
        return ""


def add_setting(df):
    conditions = [
        (df['serial'] == 1),
        (df['serial'] == 3),

        (df['serial'] == 4),

        (df['serial'] == 5),
        (df['serial'] == 6),

        (df['serial'] == 7),
        (df['serial'] == 8),
        (df['serial'] == 9),

        (df['serial'] == 10),
        (df['serial'] == 11),
        (df['serial'] == 12),

        (df['serial'] == 14),
        (df['serial'] == 15),
        (df['serial'] == 16),
        (df['serial'] == 17),
        (df['serial'] == 18),

        (df['serial'] == 20),

        (df['serial'] == 21),
        (df['serial'] == 22),
        (df['serial'] == 23),
        (df['serial'] == 24),
        (df['serial'] == 25),

        (df['serial'] == 27),

        (df['serial'] == 40),
        (df['serial'] == 41),

        (df['serial'] == 42),
        (df['serial'] == 43),

        (df['serial'] == 61),

        (df['serial'] == 71),
        (df['serial'] == 81),

        (df['serial'] == 90),
        (df['serial'] == 91),
        (df['serial'] == 92),
        (df['serial'] == 93),

        (df['serial'] == 100),
        (df['serial'] == 101),
        (df['serial'] == 102),
        (df['serial'] == 103),

        (df['serial'] == 210),
        (df['serial'] == 211),
        (df['serial'] == 220),
        (df['serial'] == 221),

        (df['serial'] == 230),
        (df['serial'] == 231),
        (df['serial'] == 240),
        (df['serial'] == 241),

        (df['serial'] == 250),
        (df['serial'] == 251),

        (df['serial'] == 260),
        (df['serial'] == 261),

    ]

    df['setting'] = np.select(conditions, SERIALS_EXPLANATIONS, default='')
    return df


def load_df(input_file):
    df = pd.read_csv(input_file)

    # methods
    df = df.fillna({'classifier': '', 'selector': '', 'adapter': '', 'prompt': '',
                    'ila': '', 'ila_locs': '', 'transfer_learning': ''})

    df['ila_str'] = df.apply(determine_ila_method, axis=1)
    df['classifier_str'] = df['classifier'].apply(lambda x: f'_{x}' if x else '')
    df['selector_str'] = df['selector'].apply(lambda x: f'_{x}' if x else '')
    df['adapter_str'] = df['adapter'].apply(lambda x: f'_{x}' if x else '')
    df['prompt_str'] = df['prompt'].apply(lambda x: f'_{x}' if x else '')
    df['transfer_learning_str'] = df['transfer_learning'].apply(lambda x: '_saw' if x is True else '')
    df['freeze_backbone_str'] = df['freeze_backbone'].apply(lambda x: '_fz' if x is True else '')

    df['method'] = df['model_name'] + df['ila_str'] + df['classifier_str'] + df['selector_str'] + df['adapter_str'] + df['prompt_str'] + df['transfer_learning_str'] + df['freeze_backbone_str']

    df = add_setting(df)

    df.rename(columns={'val_acc': 'acc'}, inplace=True)
    return df


def keep_columns(df, type='acc'):
    if type == 'acc':
        keep = ['acc', 'dataset_name', 'serial', 'setting', 'method']
    elif type == 'inference_cost':
        keep = ['host', 'serial', 'setting', 'dataset_name', 'method',
                'batch_size', 'throughput', 'flops', 'max_memory']
    elif type == 'train_cost':
        keep = ['host', 'serial', 'setting', 'method', 'batch_size', 'epochs',
                'dataset_name', 'num_images_train', 'num_images_val',
                'time_total', 'flops', 'max_memory',
                'no_params_trainable', 'no_params']

    df = df[keep]
    return df


def filter_df(df, keep_datasets=None, keep_methods=None, keep_serials=None,
              filter_datasets=None, filter_methods=None, filter_serials=None):
    if keep_datasets:
        df = df[df['dataset_name'].isin(keep_datasets)]

    if keep_methods:
        df = df[df['method'].isin(keep_methods)]

    if keep_serials:
        df = df[df['serial'].isin(keep_serials)]

    if filter_datasets:
        df = df[~df['dataset_name'].isin(filter_datasets)]

    if filter_methods:
        df = df[~df['method'].isin(filter_methods)]

    if filter_serials:
        df = df[~df['serial'].isin(filter_serials)]

    return df


def preprocess_df(
    input_file, type='acc', keep_datasets=None, keep_methods=None, keep_serials=None,
    filter_datasets=None, filter_methods=None, filter_serials=None):
    # load dataset and preprocess to include method and setting columns, rename val_acc to acc
    df = load_df(input_file)

    # drop columns
    df = keep_columns(df, type=type)

    # filter
    df = filter_df(df, keep_datasets, keep_methods, keep_serials,
                   filter_datasets, filter_methods, filter_serials)

    # sort
    df = sort_df(df)

    # saw results (serial 91, 101 -> 1, 3)
    # to be consistent with the naming criteria in the other files / results
    df = reassign_serial(df)

    return df


def round_combine_str_mean_std(df, col='acc'):
    df[f'{col}'] = df[f'{col}'].round(2)
    df[f'{col}_std'] = df[f'{col}_std'].round(2)

    df[f'{col}_mean_std_latex'] = "$" + df[f'{col}'].astype(str) + "\pm{" + df[f'{col}_std'].astype(str) + "}$"
    df[f'{col}_mean_std'] = df[f'{col}'].astype(str) + "+-" + df[f'{col}_std'].astype(str)

    return df


def add_all_cols_group(df, col='dataset_name'):
    subset = df.copy(deep=False)
    subset[col] = 'all'
    df = pd.concat([df, subset], axis=0, ignore_index=True)
    return df


def drop_na(df, args):
    subset = [args.x_var_name, args.y_var_name]
    if args.hue_var_name:
        subset.append(args.hue_var_name)
    if args.style_var_name:
        subset.append(args.style_var_name)
    if args.size_var_name:
        subset.append(args.size_var_name)
    df = df.dropna(subset=subset)
    return df


def sort_df(df, method_only=False):
    if method_only:
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS_DIC.keys(), ordered=True)

        df = df.sort_values(by=['serial', 'setting', 'method_order'], ascending=True)
        df = df.drop(columns=['method_order'])
    else:
        df['dataset_order'] = pd.Categorical(df['dataset_name'], categories=DATASETS_DIC.keys(), ordered=True)
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS_DIC.keys(), ordered=True)

        df = df.sort_values(by=['serial', 'setting', 'dataset_order', 'method_order'], ascending=True)
        df = df.drop(columns=['method_order', 'dataset_order'])
    return df




def group_by_family(x):
    classifiers = ('vit_b16_cls_fz', 'vit_b16_lrblp_fz', 'vit_b16_mpncov_fz',
                   'vit_b16_ifacls_fz', 'pedeit_base_patch16_224.fb_in1k_cls_fz',
                   'pedeit3_base_patch16_224.fb_in1k_cls_fz')

    pefgir = ('vit_b16_cls_psm_fz', 'vit_b16_cls_maws_fz', 'vit_b16_cal_fz',
              'vit_b16_avg_cls_rollout_fz', 'vit_b16_cls_glsim_fz')

    petl = ('vit_b16_cls_vqt_fz', 'vit_b16_cls_vpt_shallow_fz', 'vit_b16_cls_vpt_deep_fz',
            'vit_b16_cls_convpass_fz', 'vit_b16_cls_adapter_fz',
            'pedeit_base_patch16_224.fb_in1k_cls_adapter_fz',
            'pedeit3_base_patch16_224.fb_in1k_cls_adapter_fz')

    ufgir = ('clevit_fz', 'csdnet_fz', 'mixvit_fz', 'vit_b16_sil_fz')

    ila = ('vit_b16_ila_dso_cls_fz', 'vit_b16_ila_cls_fz', 'vit_b16_ila_dso_cls_adapter_fz',
           'vit_b16_ila_dso_cls_convpass_fz', 'vit_b16_ila_cls_adapter_fz',
           'vit_b16_ila_dso_cls_adapter_vpt_shallow_fz', 'vit_b16_ila_cls_adapter_vpt_shallow_fz',
           'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz',
           'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz')

    saw = ['vit_b16_ila_dso_cls_adapter_saw_fz',
           'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz',
           'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz']

    if x in ila:
        return 'ila'
    elif x in petl:
        return 'petl'
    elif x in pefgir:
        return 'pefgir'
    elif x in ufgir:
        return 'ufgir'
    elif x in saw:
        return 'saw'
    elif x in classifiers:
        return 'peclassifier'
    return x
