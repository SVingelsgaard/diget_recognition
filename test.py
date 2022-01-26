import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(f'own_numbers/1.png')
#img = np.invert(np.array(img))
plt.imshow(img, cmap=plt.cm.binary)
plt.show()