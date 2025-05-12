import os
import re
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

from utils import filter_df, rename_vars, drop_na


def make_plot(args, df):
    # Seaborn Style Settings
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width, # mod any of matplotlib rc system
            "figure.figsize": args.fig_size,
        })

    ax = sns.scatterplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name,
                            style=args.style_var_name, size=args.size_var_name,
                            sizes=tuple(args.sizes), legend='brief', data=df)

    if args.add_text_methods:
        for i, row in df.iterrows():

            method = row['Method']
            if args.add_text_ours_only and (('ILA' not in method) and ('DAMAS' not in method)):
                continue

            image_size = row['Setting']

            match = re.search(r'\d+', image_size)  # Finds the first sequence of digits
            if match:
                image_size = match.group()

            leg = f"{method} ({image_size})"
            ax.text(row[args.x_var_name], row[args.y_var_name], leg,
                    color='black', ha='center', va='bottom', fontsize=args.font_size)


    if args.log_scale_x:
        ax.set_xscale('log')
        plt.tick_params(axis='x', which='minor')
        ax.grid(True, which='both', axis='x')

        # ticks labels
        if args.x_ticks_labels:
            x_ticks = ax.get_xticks() if getattr(args, 'x_ticks', None) is None else args.x_ticks
            ax.set_xticks(x_ticks , labels=args.x_ticks_labels)

        formatter = LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))
        ax.get_xaxis().set_minor_formatter(formatter)

    if args.log_scale_y:
        ax.set_yscale('log')
        plt.tick_params(axis='y', which='minor')
        ax.grid(True, which='both', axis='y')

    # labels and title
    ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title, xlim=args.x_lim, ylim=args.y_lim)

    # Rotate x-axis or y-axis ticks lables
    if (args.x_rotation != None):
        plt.xticks(rotation = args.x_rotation)
    if (args.y_rotation != None):
        plt.yticks(rotation = args.y_rotation)

    # Change location of legend
    if args.hue_var_name:
        # sns.move_legend(ax, loc=args.loc_legend)
        handles, labels = ax.get_legend_handles_labels()
        hue_handles = handles[:len(df[args.hue_var_name].unique()) + 1]
        hue_labels = labels[:len(df[args.hue_var_name].unique()) + 1]
        ax.legend(hue_handles, hue_labels, loc=args.loc_legend)

    # save plot
    output_file = os.path.join(args.results_dir, f'{args.output_file}.{args.save_format}')
    plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print('Save plot to directory ', output_file)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Subset models and datasets
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('results_all', 'cost', 'cost.csv'),
                        help='filename for input .csv file')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str,
                        default=[
                            'vit_b16_cal_fz',
                            'vit_b16_sil',
                            'mixvit',
                            'clevit',
                            'csdnet',
                            'pedeit_base_patch16_224.fb_in1k_cls_fz',
                            'pedeit3_base_patch16_224.fb_in1k_cls_fz',
                            'pedeit_base_patch16_224.fb_in1k_cls_adapter_fz',
                            'pedeit3_base_patch16_224.fb_in1k_cls_adapter_fz',
                            'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz',
                            'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz',
                            'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz',
                            'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz',
                        ])
    parser.add_argument('--filter_serials', nargs='+', type=int, default=None)

    parser.add_argument('--add_text_methods', action='store_false')
    parser.add_argument('--add_text_ours_only', action='store_false')

    # Make a plot
    parser.add_argument('--log_scale_x', action='store_false')
    parser.add_argument('--log_scale_y', action='store_true')
    parser.add_argument('--x_var_name', type=str, default='flops',
                        help='name of the variable for x')
    parser.add_argument('--y_var_name', type=str, default='acc',
                        help='name of the variable for y')
    parser.add_argument('--hue_var_name', type=str, default='family',
                        help='legend of this bar plot')
    parser.add_argument('--style_var_name', type=str, default=None,
                        help='legend of this bar plot')
    parser.add_argument('--size_var_name', type=str, default='trainable_percent',)
    parser.add_argument('--orient', type=str, default=None,
                        help='orientation of plot "v", "h"')

    # output
    parser.add_argument('--output_file', default='acc_vs_flops_vs_params_log', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'plots'),
                        help='The directory where results will be stored')
    parser.add_argument('--save_format', choices=['pdf', 'png', 'jpg'], default='png', type=str,
                        help='Print stats on word level if use this command')

    # style related
    parser.add_argument('--context', type=str, default='notebook',
                        help='''affects font sizes and line widths
                        # notebook (def), paper (small), talk (med), poster (large)''')
    parser.add_argument('--style', type=str, default='whitegrid',
                        help='''affects plot bg color, grid and ticks
                        # whitegrid (white bg with grids), 'white', 'darkgrid', 'ticks'
                        ''')
    parser.add_argument('--palette', type=str, default='colorblind',
                        help='''
                        color palette (overwritten by color)
                        # None (def), 'pastel', 'Blues' (blue tones), 'colorblind'
                        # can create a palette that highlights based on a category
                        can create palette based on conditions
                        pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}
                        pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
                        ''')
    parser.add_argument('--color', type=str, default=None)
    parser.add_argument('--font_family', type=str, default='serif',
                        help='font family (sans-serif or serif)')
    parser.add_argument('--font_size', type=int, default=16)
    parser.add_argument('--font_scale', type=float, default=1.6,
                        help='adjust the scale of the fonts')
    parser.add_argument('--bg_line_width', type=int, default=0.25,
                        help='adjust the scale of the line widths')
    parser.add_argument('--line_width', type=int, default=0.75,
                        help='adjust the scale of the line widths')
    parser.add_argument('--fig_size', nargs='+', type=float, default=[16, 8],
                        help='size of the plot')
    parser.add_argument('--sizes', type=int, nargs='+', default=[160, 6400])
    parser.add_argument('--marker', type=str, default='o',
                        help='type of marker for line plot ".", "o", "^", "x", "*"')
    parser.add_argument('--dpi', type=int, default=300)

    # Set title, labels and ticks
    parser.add_argument('--title', type=str,
                        default='Acc. over 10 UFGIR Datasets vs FLOPs with Size Proportional to Task-Specific Parameters',
                        help='title of the plot')
    parser.add_argument('--x_label', type=str, default='Inference FLOPs (10^9)',
                        help='x label of the plot')
    parser.add_argument('--y_label', type=str, default='Mean Accuracy (%)',
                        help='y label of the plot')
    parser.add_argument('--x_lim', nargs='*', type=int, default=None,
                        help='limits for x axis (use if log_scale_x)')
    parser.add_argument('--y_lim', nargs='*', type=int, default=None,
                        help='limits for y axis (suggest --ylim 0 100)')
    parser.add_argument('--x_ticks', nargs='+', type=int, default=None)
    parser.add_argument('--x_ticks_labels', nargs='+', type=str, default=None,
                        help='labels of x-axis ticks')
    parser.add_argument('--x_rotation', type=int, default=None,
                        help='lotation of x-axis lables')
    parser.add_argument('--y_rotation', type=int, default=None,
                        help='lotation of y-axis lables')

    # Change location of legend
    parser.add_argument('--loc_legend', type=str, default='lower right',
                        help='location of legend options are upper, lower, left right, center')

    args= parser.parse_args()
    return args


def preprocess_df(args):
    df = pd.read_csv(args.input_file)

    df = filter_df(
        df,
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
    )

    df = drop_na(df, args)

    df = rename_vars(df, var_rename=True, args=args)
    return df


def main():
    args = parse_args()
    args.title = args.title.replace("\\n", "\n")
    os.makedirs(args.results_dir, exist_ok=True)

    if args.color:
        # single color for whole palette (sns defaults to 6 colors)
        args.palette = [args.color for _ in range(len(args.subset_models))]

    df = preprocess_df(args)

    make_plot(args, df)

    return 0

if __name__ == '__main__':
    main()