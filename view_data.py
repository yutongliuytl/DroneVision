import numpy as np 
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
import cv2

img_path = "./data/test.jpg"
img_array = cv2.imread(img_path, -1)

IMG_SIZE =128

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# cv2.imshow('test', new_array)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

new_array2 = new_array.copy()
new_array2[:, :, 0] = new_array[:, :, 2]
new_array2[:, :, 2] = new_array[:, :, 0]
plt.imshow(new_array2)
plt.show()