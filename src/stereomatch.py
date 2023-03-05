#!/usr/bin/env python
# --------------------------------------------------------------------
# Simple sum of squared differences (SSD) stereo-matching using Numpy
# --------------------------------------------------------------------


# Copyright (c) 2016 David Christian
# Modified by Heloísa Oss Boll and Lucas Ceschini
# Licensed under the MIT License

import numpy as np
from numba import jit


class stereo_match:
    def __init__(self, left_image, right_image, cost, matcher, max_offset, kernel):
        self.left_img = left_image
        self.right_img = right_image
        self.cost = cost
        self.matcher = matcher
        self.max_offset = max_offset
        self.kernel = kernel
        self.cost_volume = None
        self._result = None
        return

    def compute(self):
        self.cost_volume = self.cost.compute(
            self.left_img, self.right_img, self.max_offset, self.kernel)
        self._result = self.matcher.match(self.cost_volume)
        return

    def result(self):
        return self._result


class stereo_match_old:
    @jit(nopython=True, parallel=True, cache=True)
    def compute(left_img, right_img, shape, kernel, max_offset, cost):
        # opencv shape = (h,w,c)
        h, w, _ = shape  # assume that both images are same size

        # Depth (or disparity) map
        depth = np.zeros((w, h), np.uint8)
        # depth.shape = h, w

        kernel_half = int(kernel / 2)
        # this is used to map depth map output to 0-255 range
        offset_adjust = 255 / max_offset

        for y in range(kernel_half, h):

            for x in range(kernel_half, w - kernel_half):
                best_offset = 0
                prev_ssd = np.zeros((w, h), np.int64)
                # prev_ssd += 65534
                for _, i in enumerate(prev_ssd):
                    prev_ssd[i] = 65534

                for offset in range(max_offset):
                    ssd = np.zeros((w, h), np.int64)

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
                            if cost == 'ssd':
                                diff = ((left_img[y+v, x+u]) -
                                        (right_img[y+v, (x+u) - offset])**2)
                                ssd = ssd + diff

                            elif cost == 'sad':
                                diff = (
                                    np.abs((left_img[y+v, x+u]) - (right_img[y+v, (x+u) - offset])))
                                ssd = ssd + diff

                    # if this value is smaller than the previous ssd block
                    # then it's theoretically a closer match.
                    # Store this value against this block..
                    if ssd < prev_ssd:
                        prev_ssd = ssd
                        best_offset = offset

                # set depth output for this x,y location to the best match
                depth[y, x] = best_offset * offset_adjust
        return depth
