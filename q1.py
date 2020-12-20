import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[1, 2, 0, 1], [1, 1, 3, 2], [0, 2, 2, 1], [1, 2, 3, 4]])
kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])
kernel_size = 3


def conv_op(original, original_size, kernel, kernel_size, stride, padding):
    top_size = (original_size + 2*padding - kernel_size)//stride + 1
    top = np.zeros(shape=(top_size, top_size))
    for h in range(top_size):
        for w in range(top_size):
            top[h, w] = np.sum(original[h*stride: h*stride+kernel_size,
                                        w*stride: w*stride+kernel_size] * kernel)
    print(top)
    return top


if __name__ == '__main__':
    img = cv.imread('test.png')
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    conv_img = conv_op(b, 200, kernel1, 3, 1, 0)
    cv.imshow("orignal", img)
    cv.imshow("test", conv_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
