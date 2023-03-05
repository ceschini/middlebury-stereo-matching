import cv2
import os
import numpy as np
from PIL import Image


class IO:

    @staticmethod
    def import_image(file_name, color_space='bgr'):
        img = cv2.imread(file_name)
        shape = img.shape
        if color_space == 'lab':
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, _, _ = cv2.split(img_lab)
            return np.asarray(l), shape

        elif color_space == 'gray':
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img_gray, shape
        return img, shape

    @staticmethod
    def export_image(image, dir, name):
        if dir is None:
            dir = ""
        elif not os.path.isdir(dir):
            os.mkdir(dir)

        path = os.path.join(dir, name)
        file_name = path + '.png'
        print(f'saving image to {file_name}')
        Image.fromarray(image).save(file_name)
        return file_name


class metrics:

    def __init__(self):
        self.mse = None

    def mse(imageA, imageB):
        # ! both imgs must have same dims
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # lower the error, the more similar
        return err

    def log_metrics(mse, bad_pix, k, name, cost):
        log = open("log.txt", "a")
        log.write(f"Metrics of {name}, kernel {k} and {cost} function:\n")
        log.write(f"MSE: {mse}\n")
        log.write(f"Bad pixels: {bad_pix}%\n")
        log.close()
        return 1
