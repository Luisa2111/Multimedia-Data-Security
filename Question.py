import os
from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from math import sqrt

if not os.path.isfile('lena.bmp'):
  print("ERROR lena.bmp not found")
  #!wget -O lena.bmp "https://drive.google.com/uc?export=download&id=17MVOgbOEOqeOwZCzXsIJMDzW5Dw9-8zm"
img_path='lena.bmp'
np.random.seed(seed=123)

#Da qui inizia il codice

image = cv2.imread(img_path, 0)
im2 = image.copy()
im = np.asarray(image,dtype=np.uint8)
attacked = np.asarray(im2,dtype=np.uint8)

for i in range(100,200):
  for j in range(100, 200):
    attacked[i, j] = 1
#print(attacked)
#print(im)
def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0

  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  csf = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels

print(wpsnr(im,attacked))

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title('Original')
plt.imshow(im, cmap='gray')
plt.subplot(122)
plt.title('Watermarked')
plt.imshow(attacked, cmap='gray')
plt.show()
