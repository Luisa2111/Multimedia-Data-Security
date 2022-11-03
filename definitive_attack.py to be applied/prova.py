import numpy as np
from scipy.fft import dct, idct
import hvs_lambda as hvs
import cv2
import matplotlib.pyplot as plt

number_image = input("Enter number from 0000 to 0100:")
string_image="./sample_images_roc/"+number_image+".bmp"
image = cv2.imread(string_image, 0)

plt.figure(figsize=(20, 20))

plt.subplot(231)
plt.title('Original')
plt.imshow(image, cmap='gray')

plt.subplot(232)
plt.title('Matrix')
matrix = hvs.hvs_quantization_Xi(image)
plt.imshow(matrix, cmap='gray')

plt.subplot(233)
plt.title('Mean')
hvs = hvs.hvs_blocks(image)
print('max',np.max(hvs),'| != 0', np.count_nonzero(hvs))
print(hvs)
plt.imshow(hvs, cmap='gray')

plt.subplot(234)
plt.title('dct')
dct_image=dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')
plt.imshow(np.log(np.abs(dct_image)), cmap='gray')

plt.subplot(235)
plt.title('inverse with changes in high frequences')
changed_high=dct_image.copy()
changed_high[len(changed_high)-100:len(changed_high), len(changed_high)-100:len(changed_high)]=0
returned_as_image=idct(idct(changed_high, axis=1, norm='ortho'), axis=0, norm='ortho')
plt.imshow(returned_as_image, cmap='gray')

plt.subplot(236)
plt.title('inverse with changes in low frequences')
changed_low=dct_image.copy()
changed_low[0:100, 0:100]=0
returned_as_image=idct(idct(changed_low, axis=1, norm='ortho'), axis=0, norm='ortho')
plt.imshow(returned_as_image, cmap='gray')

plt.show()


