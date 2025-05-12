import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_contrastive_learning(df, args, mean=1, pos_color='green', neg_color='yellow'):    
    # Seaborn Style Settings
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width, # mod any of matplotlib rc system
            "figure.figsize": args.fig_size,
        })

    # Plot using seaborn
    ax = sns.scatterplot(x='x', y='y', hue='label',
                         palette={'Positive': pos_color, 'Negative': neg_color}, 
                         s=args.sizes,
                         legend='brief', data=df)

    # Add a separation line
    x = np.linspace(-mean, mean, 100)
    y = -x
    plt.plot(x, y, '--', color='gray', label='Boundary')

    # Add lines connecting positive points (attracting)
    positives = df[df['label'] == 'Positive'].copy().reset_index()
    num_positives = len(positives)
    for i in range(num_positives):
        for j in range(i + 1, num_positives):
            plt.plot([positives.loc[i, 'x'], positives.loc[j, 'x']], [positives.loc[i, 'y'], positives.loc[j, 'y']], color=pos_color, alpha=0.3)

    # Add lines connecting positive points (attracting)
    negatives = df[df['label'] == 'Negative'].copy().reset_index()
    num_negatives = len(negatives)
    for i in range(num_negatives):
        for j in range(i + 1, num_negatives):
            plt.plot([negatives.loc[i, 'x'], negatives.loc[j, 'x']], [negatives.loc[i, 'y'], negatives.loc[j, 'y']], color='orange', alpha=0.3)

    # labels, title and legend
    ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title, xlim=args.x_lim, ylim=args.y_lim)
    # sns.move_legend(ax, loc=args.loc_legend)
    plt.legend(loc=args.loc_legend)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # save plot
    output_file = os.path.join(args.results_dir, f'{args.output_file}.{args.save_format}')
    plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print('Save plot to directory ', output_file)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_points', type=int, default=4)
    parser.add_argument('--mean', type=float, default=0.25)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=10)

    # output
    parser.add_argument('--output_file', default='contrastive', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'intuition'),
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
    parser.add_argument('--font_scale', type=float, default=1.7,
                        help='adjust the scale of the fonts')
    parser.add_argument('--bg_line_width', type=int, default=0.25,
                        help='adjust the scale of the line widths')
    parser.add_argument('--line_width', type=int, default=0.75,
                        help='adjust the scale of the line widths')
    parser.add_argument('--fig_size', nargs='+', type=float, default=[4, 4],
                        help='size of the plot')
    parser.add_argument('--sizes', type=int, default=400)
    parser.add_argument('--marker', type=str, default='o',
                        help='type of marker for line plot ".", "o", "^", "x", "*"')
    parser.add_argument('--dpi', type=int, default=300)

    # Set title, labels and ticks
    parser.add_argument('--title', type=str,
                        default='SupCon:\nAttract Positives\nRepel Negatives',
                        help='title of the plot')
    parser.add_argument('--x_label', type=str, default='',
                        help='x label of the plot')
    parser.add_argument('--y_label', type=str, default='',
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

def make_data_df(num_points=4, mean=1, std=0.5, seed=0):
    np.random.seed(seed)

    # Generate positive samples (green)
    # positives = np.random.normal(0, 1, (num_points, 2)) + [mean, mean]
    positives = np.random.normal(mean, std, (num_points, 2))

    # Generate negative samples (yellow)
    # negatives = np.random.normal(0, 1, (num_points, 2)) + [-mean, -mean]
    negatives = np.random.normal(-mean, std, (num_points, 2))

    # make into pandas df
    df = pd.DataFrame(np.vstack([positives, negatives]), columns=['x', 'y'])
    df['label'] = ['Positive'] * len(positives) + ['Negative'] * len(negatives)

    return df


def main():
    args = parse_args()
    args.title = args.title.replace("\\n", "\n")
    os.makedirs(args.results_dir, exist_ok=True)

    df = make_data_df(args.num_points, args.mean, args.std, args.seed)

    # Example usage
    plot_contrastive_learning(df, args, args.mean)

    return 0

if __name__ == '__main__':
    main()
