import os
from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from math import sqrt

if not os.path.isfile('lena.bmp'):  
  !wget -O lena.bmp "https://drive.google.com/uc?export=download&id=17MVOgbOEOqeOwZCzXsIJMDzW5Dw9-8zm"
img_path='lena.bmp'
np.random.seed(seed=123)

#Da qui inizia il codice

image = cv2.imread(img_path, 0)
im = np.asarray(image,dtype=np.uint8)
attacked=im.copy()
print(attacked[0,0])
attacked[0,0]=1
print(attacked)
print(im)

print(wpsnr(im,attacked))

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title('Original')
plt.imshow(im, cmap='gray')
plt.subplot(122)
plt.title('Watermarked')
plt.imshow(attacked,cmap='gray')
plt.show()
