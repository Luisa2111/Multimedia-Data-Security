import embedding_ef26420c, detection_ef26420c
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

def jpeg_compression(img, QF):
  from PIL import Image
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')
  return attacked

def blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  attacked = gaussian_filter(img, sigma)
  return attacked

mark_p = './ef26420c.npy'

mark = np.load(mark_p)
mark = mark.reshape(32,32)
img = cv2.imread('../sample-images-roc/0000.bmp', 0)
watermarked = embedding_ef26420c.embedding('../sample-images-roc/0000.bmp', './ef26420c.npy')
cv2.imwrite('watermarked.bmp',(watermarked))

#attack
attacked = jpeg_compression(watermarked, 1)
# attacked = blur(watermarked, 10)

cv2.imwrite('attacked.bmp', attacked)
# plt.imshow(attacked)
# plt.show()
#
# start = time.time()
"""for i in range(100):
  dec, wpsnr = detection_ef26420c.detection('../sample-images-roc/0000.bmp', 'watermarked.bmp',
                                            '../sample-images-roc/' + str(i).zfill(4)+ '.bmp')
# print('time consumed: ', time.time() - start)
  if dec == 1:
    print('fp in',i)
    print(dec)
    print(wpsnr)"""

print('attack')
start = time.time()
dec, wpsnr = detection_ef26420c.detection('../sample-images-roc/0000.bmp', 'watermarked.bmp', 'attacked.bmp')
print('time consumed: ', time.time() - start)
print(dec)
print(wpsnr)