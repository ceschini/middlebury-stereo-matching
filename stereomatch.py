#!/usr/bin/env python
# --------------------------------------------------------------------
# Simple sum of squared differences (SSD) stereo-matching using Numpy
# --------------------------------------------------------------------

# Copyright (c) 2016 David Christian
# Modified by Heloísa Oss Boll and Lucas Ceschini
# Licensed under the MIT License

import numpy as np
from PIL import Image
import cv2


def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = cv2.imread(left_img)
    right_img = cv2.imread(right_img)

    # convert images to CIELAB color space
    left = cv2.cvtColor(left_img, cv2.COLOR_BGR2LAB)
    right = cv2.cvtColor(right_img, cv2.COLOR_BGR2LAB)

    # grabbing L channel
    l_left, a_left, b_left = cv2.split(left)
    l_right, a_right, b_right = cv2.split(right)


    # opencv shape = (h,w,c)
    h, w, _ = left.shape  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)
    # this is used to map depth map output to 0-255 range
    offset_adjust = 255 / max_offset

    for y in range(kernel_half, h):
        print("\rProcessing.. %d%% complete" %
              (y / (h - kernel_half) * 100), end="", flush=True)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search, used to ensure a good match
                # because the squared differences of two pixels alone is not enough to go on
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # Check boundary conditions
                        if y+v >= h or x+u >= w or (x+u) - offset < 0:
                            continue

                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of LAB, so we can use Euclidean distance
                        # to calculate the distance between the two pixels
                        # diff = left[y+v, x+u] - right[y+v, (x+u) - offset]
                        diff = l_left[y+v, x+u] - l_right[y+v, (x+u) - offset]
                        ssd_temp = np.sum(diff * diff)
                        ssd += ssd_temp

                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust

    # Convert to PIL and save it
    Image.fromarray(depth).save('img/outputs/depth.png')


if __name__ == '__main__':
    # 6x6 local search kernel, 30 pixel search range
    stereo_match("img/inputs/view0.png", "img/inputs/view1.png", 6, 30)
