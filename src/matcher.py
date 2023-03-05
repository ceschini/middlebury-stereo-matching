import numpy as np


class matcher:

    @staticmethod
    def match(cost_volume):
        return np.argmin(cost_volume, axis=2)
