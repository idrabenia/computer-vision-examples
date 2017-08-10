import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('mig_21.jpg')
img_mask = cv2.imread('mig_21_g.jpg', 0)
mask = np.full(img.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

mask[img_mask == 0] = cv2.GC_BGD
mask[img_mask == 255] = cv2.GC_FGD

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 600, 400)
mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

mask2 = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

cv2.imshow('title', cv2.resize(img, (600, 450)))

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()