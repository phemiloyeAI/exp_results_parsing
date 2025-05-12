import os
import glob
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


DATASETS_DIC = {
    'cotton': 'Cotton',
    'soyageing': 'SoyAgeing',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',

    'leaves': 'Leaves',
}


def search_images(folder):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    images_path = []
    for file_type in types:
        # images_path is the list of files
        path = os.path.join(folder, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        images_path.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files pre-filtering', len(images_path))

    return images_path


def filter_images(images_path, serial=1, datasets=['_'], test_images=False):
    filtered = set()
    for file in images_path:
        if f'_{serial}' in file:
            if any([ds in file for ds in datasets]):
                if test_images and 'test' in file:
                    filtered.add(file)
                elif 'train' in file:
                    filtered.add(file)

    filtered = sorted(filtered)
    print('Images after filtering: ', len(filtered), filtered)
    return filtered


def load_resize_imgs(images_paths, image_size=224):
    imgs_all = []

    for fp in images_paths:
        img = Image.open(fp)

        width, height = img.size
        if width >= height:
            r = width / height
            new_h = image_size
            new_w = int(r * image_size)
        else:
            r = height / width
            new_w = image_size
            new_h = int(r * image_size)
        img = img.resize((new_w, new_h))

        # PIL images use shape w, h but NP uses h, w
        img_np = np.array(img)
        imgs_all.append(img_np)

    return imgs_all


def make_img_grid(args):
    images_paths = search_images(args.input_folder)

    images_paths = filter_images(images_paths, serial=args.serial,
                                 datasets=args.datasets)

    imgs_all = load_resize_imgs(images_paths, args.image_size)


    number_imgs = len(imgs_all)


    if args.dataset_classes:
        # one row with multiple dataset-classes
        fig = plt.figure(figsize=(args.number_images_per_ds * number_imgs, 1))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, number_imgs),
                        axes_pad=(0.1, 0.0), direction='row', aspect=True)
    else:
        # one dataset per row
        # width, height
        fig = plt.figure(figsize=(args.number_images_per_ds, number_imgs))
        grid = ImageGrid(fig, 111, nrows_ncols=(number_imgs, 1),
                        axes_pad=(0.0, 0.0), direction='row', aspect=True)


    for i, (ax, np_arr) in enumerate(zip(grid, imgs_all)):
        # ax.axis('off')
        ax.imshow(np_arr)

        ax.tick_params(top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        [ax.spines[side].set_visible(False) for side in ('top', 'right', 'bottom', 'left')]

        if args.label_datasets:
            if args.dataset_classes:
                # last row
                    label = args.dataset_classes_labels[i]
                    print(i, label)
                    ax.xaxis.set_visible(True)
                    ax.set_xlabel(label, fontsize=args.font_size_title)
            else:
                # first column in each row
                    label = DATASETS_DIC[args.datasets[i]]
                    print(i, label)
                    ax.yaxis.set_visible(True)
                    ax.set_ylabel(label, fontsize=args.font_size_title)


    # fig.tight_layout()
    fig.savefig(args.output_file, dpi=args.dpi, bbox_inches='tight', pad_inches=0.01)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_folder', type=str,
                        default=os.path.join('data', 'results_inference'),
                        help='name of folder which contains the results')

    # filtering
    parser.add_argument('--serial', type=int, default=420,
                        help='serial for images (def: 1 for vertical datasets with 2 imgs)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cotton', 'soyageing', 'soygene', 'soyglobal', 'soylocal'],
                        help='names of datasets for labeling')
    parser.add_argument('--test_images', action='store_true')

    # resizing and visualizing format
    parser.add_argument('--number_images_per_ds', type=int, default=12)
    parser.add_argument('--image_size', default=224, type=int, help='file size')

    parser.add_argument('--dataset_classes', action='store_true',
                        help='by def uses dataset name')
    parser.add_argument('--label_datasets', action='store_false')
    parser.add_argument('--dataset_classes_labels', type=str, nargs='+',
                        default=['SoyGene Class 1', 'SoyGene Class 2', 'SoyLocal Class 1', 'SoyLocal Class 2'],
                        help='list of labels when using dataset_classes')

    # output file
    parser.add_argument('--save_name', default='datasets_ufgir',
                        type=str, help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'datasets'),
                        help='The directory where results will be stored')

    parser.add_argument('--save_format', choices=['pdf', 'png', 'jpg'], default='png', type=str,
                        help='Print stats on word level if use this command')
    parser.add_argument('--font_size_title', type=int, default=12)
    parser.add_argument('--dpi', type=int, default=300)

    args= parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = f'{os.path.join(args.results_dir, args.save_name)}.{args.save_format}'

    make_img_grid(args)

    return 0


if __name__ == '__main__':
    main()
