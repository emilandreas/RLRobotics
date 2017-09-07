import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def cropImg(img,n, m):
    topCrop = 32

    realN = img.__len__()
    realM = img[0].__len__()
    if n <= realN-topCrop and m <= realM:
        return img[topCrop:topCrop+n, :m]
    else:
        print('ERROR CROPPING IMAGE.')

def displayGray(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()