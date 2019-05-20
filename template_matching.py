import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# img = cv.imread('./pictures/brandworkz/Brandworkz-Logo-5.png', 0)
# template = cv.imread('./pictures/brandworkz/Brandworkz-Logo-3.png', 0)

img = cv.imread('./pictures/brandworkz/Brandworkz-Logo-5.png', 0)
img2 = img.copy()
template = cv.imread('./pictures/brandworkz/Brandworkz-Logo-3.png', 0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    plt.imshow(res, cmap='gray')
    plt.show()
