from numba import jit
import numpy as np


class ssd:
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def compute(left_img, right_img, kernel, max_offset):
        # opencv shape = (h,w,c)
        (H, W) = left_img.shape
        cost_volume = np.zeros((H, W, max_offset))

        # Loop over internal image
        for y in range(kernel, H - kernel):
            for x in range(kernel, W - kernel):
                # Loop over window
                for v in range(-kernel, kernel + 1):
                    for u in range(-kernel, kernel + 1):
                        # Loop over all possible disparities
                        for d in range(0, max_offset):
                            cost_volume[y, x, d] += (left_img[y+v,
                                                              x+u] - right_img[y+v, x+u-d])**2
        return cost_volume
