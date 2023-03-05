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

    def rmse(estimate, gt):
        # ! both imgs must have same dims
        err = np.sum((estimate.astype("float") - gt.astype("float")) ** 2)
        err /= float(estimate.shape[0] * estimate.shape[1])
        err = np.sqrt(err)

        # lower the error, the more similar
        return err

    def bad_pixels(estimate, gt):
        gt = np.round(gt / 4).astype(np.uint8)
        bad_pixels = np.sum(np.abs(gt - estimate) > 5)
        total_pixels = gt.shape[0] * gt.shape[1]
        error_rate = bad_pixels / total_pixels
        return error_rate * 100

    def log_metrics(rmse, bad_pix, k, off, name, cost):
        log = open("log.csv", "a")
        log.write(f"{name},{k},{off},{cost},{rmse},{bad_pix}\n")
        log.close()
        return 1
