#Section 5, scipy project
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
img = cv2.imread(r'C:\Users\diego\Downloads\28223971.jpg')

img_scaled = img/255
red = img[:,:, 0]
green = img[:,:,1]
blue = img[:,:,2]
plt.subplot(221)
plt.imshow(red, cmap = plt.cm.Reds_r)
plt.subplot(222)
plt.imshow(blue, cmap = plt.cm.Blues_r)
plt.subplot(223)
plt.imshow(green, cmap = plt.cm.Greens_r)
plt.subplot(224)
plt.imshow(img)
plt.show()
