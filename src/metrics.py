import numpy as np


def mse(imageA, imageB):
    # ! both imgs must have same dims
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # lower the error, the more similar
    return err
