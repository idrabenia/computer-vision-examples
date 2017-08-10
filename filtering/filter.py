import cv2
import numpy as np


def filter2d_probe(gray):
    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[1][1] = 2
    kernel = kernel - np.ones((3, 3), dtype=np.float32) * 1 / 9
    return cv2.filter2D(gray, -1, kernel)


img = cv2.imread('mig_21.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imwrite('filter2d.jpg', filter2d_probe(img))
cv2.imwrite('mean_filter.jpg', cv2.medianBlur(img, ksize=15))
cv2.imwrite('sobel.jpg', cv2.Sobel(gray, -1, 0, 1))
cv2.imwrite('laplasian.jpg',
            cv2.threshold(cv2.Laplacian(cv2.medianBlur(gray, ksize=9), -1), 10, 200,
                          cv2.THRESH_BINARY)[1]
            )

cv2.imwrite('canny.jpg', cv2.Canny(img, 10, 255))
