#!/usr/bin/env python
# --------------------------------------------------------------------
# Simple sum of squared differences (SSD) stereo-matching using Numpy
# --------------------------------------------------------------------


# Copyright (c) 2016 David Christian
# Modified by Heloísa Oss Boll and Lucas Ceschini
# Licensed under the MIT License

import argparse
from src.utils import metrics, IO
from src.stereomatch import stereo_match
from src.matcher import matcher
from src.ssd import ssd
import matplotlib.pyplot as plt


def main(left_img, right_img, gt, kernel, max_offset, cost='ssd'):
    # Load in both images, assumed to be RGBA 8bit per channel images
    gt_name = gt.split('/')[-1].split('.')[0]
    img_name = left_img.split('/')[-1].split('.')[0]

    left_img = IO.import_image(left_img, 'gray')
    # l, a, b = IO.split_channels(left_img)
    left_l = left_img
    right_img = IO.import_image(right_img, 'gray')
    # l, a, b = IO.split_channels(right_img)
    right_l = right_img
    gt = IO.import_image(gt)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1), plt.imshow(
        left_img, cmap='gray'), plt.title('Left')
    plt.subplot(1, 2, 2), plt.imshow(
        right_img, cmap='gray'), plt.title('Right')
    plt.tight_layout()

    matching_alg = matcher
    if cost == 'ssd':
        matching_cost = ssd

    sm = stereo_match(left_l, right_l, matching_cost,
                      matching_alg, max_offset, kernel)

    print('Processing...')
    sm.compute()
    print('Completed')
    depth = sm.result()
    plt.figure()
    plt.imshow(depth, cmap='gray')
    plt.show()

    # save it
    IO.export_image(IO.normalize_image(depth, gt), 'img/outputs/',
                    f'{img_name}_{gt_name}_{kernel}x{kernel}')

    # process metrics and log it
    metrics.log_metrics(depth, gt, kernel, img_name, cost)


if __name__ == '__main__':
    # 3x3 local search kernel, 60 pixel search range by default
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--left', type=str, help='Path to left image')
    parser.add_argument('-r', '--right', type=str, help='Path to right image')
    parser.add_argument('-c', '--cost', type=str, choices=['ssd', 'sad'],
                        help='Cost function', default='ssd')
    parser.add_argument('-gt', '--ground_truth', type=str,
                        help='Path to ground truth image')
    parser.add_argument('-k', '--kernel', type=int,
                        help='kernel window size', default=3)
    parser.add_argument('-o', '--offset', type=int, default=60,
                        help='maximum offset for pixel search range')
    args = parser.parse_args()
    main(args.left, args.right, args.ground_truth,
         args.kernel, args.offset, args.cost)
