import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lena_gray.jpg')
# opencv imread w/o flag will defaultly read image as colored
#img = plt.imread('lena_gray.jpg')
print(img.shape)