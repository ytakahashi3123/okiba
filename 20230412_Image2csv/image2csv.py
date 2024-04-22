#!/usr/bin/env python3

import numpy as np
from PIL import Image
import cv2 as cv2

# Read file
image = np.array(Image.open('クラーク0.jpg').convert('L'))
print(image.shape)

# Gaussian filter
gauss_kernel = 9
gauss_sdv    = 0
gauss = cv2.GaussianBlur(image, (gauss_kernel, gauss_kernel), gauss_sdv)
#cv2.imshow('gauss', gauss)
cv2.imwrite('gauss.jpg', gauss)

# Output data in CSV
# --Original
np.savetxt('np_savetxt.csv', image, delimiter=',', fmt='%.5e')
# --Gaussian
np.savetxt('np_savetxt_gauss.csv', gauss, delimiter=',', fmt='%.5e')
