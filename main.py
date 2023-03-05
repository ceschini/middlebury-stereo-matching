#!/usr/bin/env python
# --------------------------------------------------------------------
# Simple sum of squared differences (SSD) stereo-matching using Numpy
# --------------------------------------------------------------------


# Copyright (c) 2016 David Christian
# Modified by Heloísa Oss Boll and Lucas Ceschini
# Licensed under the MIT License

import argparse
from src.utils import IO
from src.utils import metrics
from src.stereomatch import stereo_match
from src.ssd import ssd
import cv2


def main(left_image, right_image, gt, kernel, max_offset, cost='ssd'):
    # Load in both images, assumed to be RGBA 8bit per channel images
    gt_name = gt.split('/')[-1].split('.')[0]
    img_name = left_image.split('/')[-1].split('.')[0]

    left_img, shape = IO.import_image(left_image, 'lab')
    right_img, _ = IO.import_image(right_image, 'lab')
    gt, _ = IO.import_image(gt, 'gray')

    cv2.imshow('Left', left_img)
    cv2.imshow('Right', right_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if cost == 'ssd':
        cost_function = ssd
    sm = stereo_match(left_img, right_img, cost_function,
                      max_offset, kernel, shape)

    print('Processing...')
    depth = sm.compute()
    print('Completed')
    cv2.imshow('depth', depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save it
    IO.export_image(depth, 'img/outputs/',
                    f'{img_name}_{gt_name}_{kernel}x{kernel}')

    # process metrics and log it
    mse = metrics.mse(depth, gt)
    metrics.log_metrics(mse, kernel, img_name, cost)


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
                        help='kernel window size', default=6)
    parser.add_argument('-o', '--offset', type=int, default=30,
                        help='maximum offset for pixel search range')
    args = parser.parse_args()
    main(args.left, args.right, args.ground_truth,
         args.kernel, args.offset, args.cost)
