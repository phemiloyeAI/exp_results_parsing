import os
import glob
import argparse

import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


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
                if test_images and 'test' in file and (('None' in file)):
                    filtered.add(file)
                elif 'train' in file and (('None' in file)):
                    filtered.add(file)

    filtered = sorted(filtered)
    print('Images after filtering: ', len(filtered), filtered)
    return filtered


def find_substring_in_list(string, substr_list):
    """
    Check if any substring from substr_list is present in string.
    If found, return the first matching substring; otherwise, return None.
    """
    for substr in substr_list:
        if substr in string:
            return substr
    raise 

def get_vis_paths(file, dataset_list, model_list, vis_list, test_images=False):
    paths = []

    fp, fn = os.path.split(file)
    fp, ds_mn_serial = os.path.split(fp)

    # ds = ds_mn_serial.split('_')[0]
    ds = find_substring_in_list(ds_mn_serial, dataset_list)

    for model, vis in zip(model_list, vis_list):
        ds_mn = f'{ds}_{model}'
        split = 'test' if test_images else 'train'
        full_fn = os.path.join(fp, ds_mn, f'{split}_{vis}.png')
        paths.append(full_fn)

    print('Grid images paths: ', paths)
    return paths


def load_resize_imgs(images_paths, args):
    imgs_all = []

    for file in images_paths:
        full_names = get_vis_paths(file, args.datasets, args.models, args.vis_types, args.test_images)

        imgs = [Image.open(fp) for fp in full_names]
        width, height = imgs[0].size
        if width >= height:
            r = width / height
            new_h = args.image_size
            new_w = int(r * args.image_size)

        else:
            r = height / width
            new_w = args.image_size
            new_h = int(r * args.image_size)

        imgs = [img.resize((new_w, new_h)) for img in imgs]

        # PIL images use shape w, h but NP uses h, w
        imgs_np = [np.array(img) for img in imgs]
        imgs_all.extend(imgs_np)

    return imgs_all


def make_img_grid(args):
    images_paths = search_images(args.input_folder)

    images_paths = filter_images(images_paths, serial=args.serial,
                                 datasets=args.datasets)


    imgs_all = load_resize_imgs(images_paths, args)


    vis_labels_list = args.vis_labels
    datasets_list = args.datasets

    number_imgs = len(images_paths)
    number_vis = len(args.vis_types)


    fig = plt.figure(figsize=(number_imgs * args.number_images_per_ds, number_vis))
    grid = ImageGrid(fig, 111, nrows_ncols=(number_vis, number_imgs),
                     axes_pad=(0.01, 0.01), direction='row', aspect=True)


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

        # first column in each row
        if args.label_vis and ((i + 1) % number_imgs == 1 or (number_imgs == 1)):
            label = vis_labels_list[i // number_imgs]
            print(i, label)
            ax.yaxis.set_visible(True)
            ax.set_ylabel(label, fontsize=args.font_size_title)

        # last row
        if args.label_datasets and ((i + 1) / number_imgs > (number_vis - 1)):
            label = datasets_list[i % number_imgs]
            print(i, label)
            ax.xaxis.set_visible(True)
            ax.set_xlabel(label, fontsize=args.font_size_title)


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
    parser.add_argument('--serial', type=int, default=30,
                        help='serial for images')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['soyageing'],
                        help='names of datasets for labeling')
    parser.add_argument('--test_images', action='store_true')

    parser.add_argument('--models', type=str, nargs='+',
                        default=[
                            'vit_b16_cls_fz_30',

                            'vit_b16_cls_fz_30',
                            'vit_b16_cls_adapter_adapter_30',
                            'vit_b16_ila_dso_cls_adapter_ila_30',
                            'vit_b16_ila_dso_cls_adapter_saw_30',
                            'vit_b16_ila_dso_cls_adapter_saw_ft_30',

                            'vit_b16_cls_fz_30',
                            'vit_b16_cls_adapter_adapter_30',
                            'vit_b16_ila_dso_cls_adapter_ila_30',
                            'vit_b16_ila_dso_cls_adapter_saw_30',
                            'vit_b16_ila_dso_cls_adapter_saw_ft_30',

                            'vit_b16_cls_fz_30',
                            'vit_b16_cls_adapter_adapter_30',
                            'vit_b16_ila_dso_cls_adapter_ila_30',
                            'vit_b16_ila_dso_cls_adapter_saw_30',
                            'vit_b16_ila_dso_cls_adapter_saw_ft_30',
                            ],
                        help='names of datasets for labeling')
    parser.add_argument('--vis_types', type=str, nargs='+',
                        default=[
                            'None',

                            'rollout_0_4',
                            'rollout_0_4',
                            'rollout_0_4',
                            'rollout_0_4',
                            'rollout_0_4',

                            'rollout_4_8',
                            'rollout_4_8',
                            'rollout_4_8',
                            'rollout_4_8',
                            'rollout_4_8',

                            'rollout_8_12',
                            'rollout_8_12',
                            'rollout_8_12',
                            'rollout_8_12',
                            'rollout_8_12',
                            ],
                        help='names of datasets for labeling')
    parser.add_argument('--vis_labels', type=str, nargs='+',
                        default=[
                            'Samples',
                            'ViT (L)',
                            'ADP (L)',
                            'ILA (L)',
                            'SAW (L)',
                            'SAW+FT (L)',

                            'ViT (M)',
                            'ADP (M)',
                            'ILA (M)',
                            'SAW (M)',
                            'SAW+FT (M)',

                            'ADP (H)',
                            'ViT (H)',
                            'ILA (H)',
                            'SAW (H)',
                            'SAW+FT (H)',
                            ],
                        help='names of datasets for labeling')


    # resizing and visualizing format
    parser.add_argument('--number_images_per_ds', type=int, default=12)
    parser.add_argument('--image_size', default=224, type=int, help='file size')

    parser.add_argument('--label_datasets', action='store_true')
    parser.add_argument('--label_vis', action='store_false')

    # output file
    parser.add_argument('--save_name', default='rollout_all_soyageing_all',
                        type=str, help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'vis'),
                        help='The directory where results will be stored')

    parser.add_argument('--save_format', choices=['pdf', 'png', 'jpg'],
                        default='png', type=str,
                        help='Print stats on word level if use this command')
    parser.add_argument('--font_size_title', type=int, default=12)
    parser.add_argument('--dpi', type=int, default=300)


    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = f'{os.path.join(args.results_dir, args.save_name)}.{args.save_format}'

    make_img_grid(args)

    return 0


if __name__ == '__main__':
    main()
