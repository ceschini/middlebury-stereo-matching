from numba import jit
import numpy as np


class ssd:
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def compute(left_img, right_img, kernel, max_offset, shape):
        # opencv shape = (h,w,c)
        h, w, _ = shape

        # Depth (or disparity) map
        depth = np.zeros((h, w), np.uint8)
        # depth = np.zeros((w, h), np.uint8)

        kernel_half = int(kernel / 2)
        # this is used to map depth map output to 0-255 range
        offset_adjust = 255 / max_offset

        for y in range(kernel_half, h):
            for x in range(kernel_half, w - kernel_half):
                best_offset = 0
                prev_ssd = 65534

                for offset in range(max_offset):
                    ssd = 0
                    ssd_temp = 0

                    for v in range(-kernel_half, kernel_half):
                        for u in range(-kernel_half, kernel_half):
                            # Check boundary conditions
                            if y+v >= h or x+u >= w or (x+u) - offset < 0:
                                continue

                            diff = left_img[y+v, x+u] - \
                                right_img[y+v, (x+u) - offset]
                            ssd_temp = diff * diff
                            ssd += ssd_temp

                    if ssd < prev_ssd:
                        prev_ssd = ssd
                        best_offset = offset

                depth[y, x] = best_offset * offset_adjust
        return depth
