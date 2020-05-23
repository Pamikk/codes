import cv2
import numpy as np
imgGray = cv2.imread("circle.png", cv2.IMREAD_GRAYSCALE)
sobelX = cv2.Sobel(imgGray,7, 1, 0, ksize=3)
sobelX1 = cv2.convertScaleAbs(sobelX)
cv2.imshow('res',np.hstack((imgGray, sobelX, sobelX1)))
cv2.waitKey(0)
print(cv2.CV_32F)