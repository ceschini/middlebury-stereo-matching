import cv2
import os
import numpy as np
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray


class IO:

    @staticmethod
    def import_image(file_name, color_space='rgb'):
        img = imread(file_name)
        if color_space == 'lab':
            # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # return img_lab
            pass
        elif color_space == 'gray':
            img_gray = rgb2gray(img)
            return img_gray
        return img

    @staticmethod
    def export_image(image, dir, name):
        if dir is None:
            dir = ""
        elif not os.path.isdir(dir):
            os.mkdir(dir)

        path = os.path.join(dir, name)
        file_name = path + '.jpg'
        print(f'saving image to {file_name}')
        imsave(file_name, img_as_ubyte(image), quality=100)
        return file_name

    @staticmethod
    def split_channels(image):
        c1, c2, c3 = cv2.split(image)
        return c1, c2, c3

    @staticmethod
    def normalize_image(img, gt_img):
        norm_img = img
        if (np.max(gt_img) <= 0):
            pass
        else:
            norm_img = img/np.max(gt_img)

        return norm_img/np.max(norm_img)


class metrics:

    def mse(imageA, imageB):
        # ! both imgs must have same dims
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # lower the error, the more similar
        return err

    def log_metrics(self, estimate, ground_truth, k, name, cost):
        mserr = self.mse(estimate, ground_truth)
        log = open("../log.txt", "a")
        log.write(
            f"MSE of {name} with kernel {k} and {cost}: {mserr} \n"
        )
        log.close()
        return 1
