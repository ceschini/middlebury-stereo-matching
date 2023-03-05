#!/usr/bin/env python
# --------------------------------------------------------------------
# Simple sum of squared differences (SSD) stereo-matching using Numpy
# --------------------------------------------------------------------


# Copyright (c) 2016 David Christian
# Modified by Heloísa Oss Boll and Lucas Ceschini
# Licensed under the MIT License


class stereo_match:
    def __init__(self, left, right, cost, max_offset, kernel, shape):
        self.left_img = left
        self.right_img = right
        self.cost = cost
        self.max_offset = max_offset
        self.kernel = kernel
        self.cost_volume = None
        self.shape = shape
        return

    def compute(self):
        return self.cost.compute(self.left_img, self.right_img,
                                 self.max_offset, self.kernel,
                                 self.shape)
