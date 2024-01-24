import os
import numpy as np
import scipy.io as scio


def another_dice(x, y):
    smooth = 1e-7
    xflat = x.flatten()
    yflat = y.flatten()
    d = []
    s = xflat * (xflat == yflat)
    # s = np.delete(s, np.argwhere(s == 0))
    inter = np.linalg.norm(s)
    union = np.linalg.norm(yflat) + np.linalg.norm(xflat)
    d = (2. * inter + smooth) / (union + smooth)
    return d

def ComputeDice(x, y):
    # x = x / 255.
    # y = y / 255.
    x = np.array(x)
    y = np.array(y)
    smooth = 1e-7
    xflat = x.flatten()
    yflat = y.flatten()
    # yflat[yflat > 0.4] = 1
    inter = np.sum(xflat * yflat)
    union = np.sum(yflat) + np.sum(xflat)
    d = (2 * inter + smooth) / (union + smooth)
    return d


def numel(S):
    '''元素个数'''
    size = S.shape
    sum = 1
    for s in size:
        sum *= s
    return sum


if __name__ == '__main__':
    import cv2
    I1 = cv2.imread('regout\models_seg1\seg_moving.jpg')
    I2 = cv2.imread('regout\models_seg1\seg_fixed.jpg')
    # I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    # I2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
    I1 = np.array(I1) / 255.
    I2 = np.array(I2) / 255.
    # d = compute_dice(I1, I2)
    # print(d)
    d = ComputeDice(I1, I2)
    print(d)
