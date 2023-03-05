#!/usr/bin/env python
# --------------------------------------------------------------------
# Simple sum of squared differences (SSD) stereo-matching using Numpy
# --------------------------------------------------------------------

# SAD - robust metric
# https://www.baeldung.com/cs/disparity-map-stereo-vision

# Copyright (c) 2016 David Christian
# Modified by Heloísa Oss Boll and Lucas Ceschini
# Licensed under the MIT License

import cv2
import numpy as np
from PIL import Image


def mse(imageA, imageB):
    # ! both imgs must have same dims
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # lower the error, more similar the images are
    return err


def log_metrics(estimate, ground_truth, k, name, cost):
    mserr = mse(estimate, ground_truth)
    log = open("../log.txt", "a")
    log.write(
        f"MSE of {name} with kernel {k} and {cost} cost function : {mserr} \n"
    )
    log.close()
    return 1


def stereo_match(left_img, right_img, gt, kernel, max_offset, cost='ssd'):
    # Load in both images, assumed to be RGBA 8bit per channel images
    gt_name = gt
    img_name = left_img
    left_img = cv2.imread(f'../img/inputs/{left_img}')
    right_img = cv2.imread(f'../img/inputs/{right_img}')
    gt = cv2.imread(f'../img/inputs/{gt}')

    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    # convert images to CIELAB color space
    left = cv2.cvtColor(left_img, cv2.COLOR_BGR2LAB)
    right = cv2.cvtColor(right_img, cv2.COLOR_BGR2LAB)

    # splitting the image and grabbing the L channel
    l_left, _, _ = cv2.split(left)
    l_right, _, _ = cv2.split(right)

    # opencv shape = (h,w,c)
    h, w, _ = left.shape  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)
    # this is used to map depth map output to 0-255 range
    offset_adjust = 255 / max_offset

    for y in range(kernel_half, h):
        print("Processing..")

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search,
                # used to ensure a good match
                # because the squared differences of two pixels alone
                #  is not enough to go on
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # Check boundary conditions
                        if y+v >= h or x+u >= w or (x+u) - offset < 0:
                            continue

                        # iteratively process differences values.

                        # left[] and right[] are arrays of LAB,
                        # so we can use Euclidean distance
                        # to calculate the distance between the two pixels

                        # diff = l_left[y+v, x+u] - l_right[y+v, (x+u) - offset]
                        diff = np.int64(l_left[y+v, x+u]) - \
                            np.int64(l_right[y+v, (x+u) - offset])

                        if cost == 'ssd':
                            ssd_temp = np.sum(diff * diff)
                            ssd += ssd_temp
                        elif cost == 'sad':
                            sad_temp = np.abs(diff)
                            ssd += sad_temp

                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match.
                # Store this value against this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust

    # Convert to PIL and save it
    output = f'{gt_name}_depth_{kernel}x{kernel}.png'
    print(f'Saving depth image to img/outputs/{output}')
    Image.fromarray(depth).save(f'../img/outputs/{output}')

    # process metrics and log it
    log_metrics(depth, gt, kernel, img_name, cost)


if __name__ == '__main__':
    # 6x6 local search kernel, 30 pixel search range
    stereo_match("im2.png", "im6.png", "disp2.png", 6, 30, "ssd")
    # stereo_match("im2.png", "im6.png", "disp2.png", 5, 30)
    # stereo_match("im2.png", "im6.png", "disp2.png", 3, 30)
