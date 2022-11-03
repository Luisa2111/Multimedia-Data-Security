import numpy as np
#from oldembeddings import embedding as em
import embedding_sub_cap_flat1 as em
import detection_sub_cap_flat1 as dt
import attack as at
from psnr import similarity, wpsnr
import cv2
import threshold_roc_curve as roc
import matplotlib.pyplot as plt

# settings
alpha = 5
dim = 8
step = 20
max_splits = 500
Xi_exp = 0.2
Lambda_exp = 0.3
L_exp = 0.2
min_splits = 170
sub_size = 6
ceil = True
# generate a watermark (in the challenge, we will be provided a mark)
mark = np.load('ef26420c.npy')
# mark = np.array([(-1) ** m for m in mark])

# embed watermark into three different pictures of 512x512 (as it will be in the challenge)
# name_image = 'sample-images-roc/0031.bmp'
img = '0000'
name_image = 'sample-images-roc/' + img +'.bmp'
name_image = "watermarking-images/original.bmp"
image = cv2.imread(name_image,0)
name_out = 'watermarking-images/watermarked.bmp'
# name_out = "watermarked.bmp"
# em.embedding(name_image, mark, name_output=name_out)

""", name_output=name_out,
             dim = dim, step = step, max_splits=max_splits,
             min_splits=min_splits, sub_size=sub_size,
             Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp , ceil = ceil"""

watermarked = cv2.imread(name_out,0)
from scipy.fft import dct, idct, fft

def im_fft(image):
    return fft(fft(image, axis=0, norm='ortho'), axis=1, norm='ortho')
def im_dct(image):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
def im_idct(image):
    return (idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho'))
import pywt

def dwt(im):
    coeffs2 = pywt.dwt2(im, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Show components
    im = im.shape
    blank_image = np.zeros((im[0], im[1]), np.float32)
    blank_image[:im[0]//2, :im[1]//2] = LL
    blank_image[:im[0]//2, im[1]//2:] = LH
    blank_image[im[0]//2:, :im[1]//2] = HL
    blank_image[im[0]//2:, im[1]//2:] = HH
    return blank_image

def dwt2(im):
    coeffs2 = pywt.dwt2(im, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Show components
    blank_image = np.zeros((512, 512), np.float32)
    blank_image[:256, :256] = dwt(LL)
    blank_image[:256, 256:] = LH
    blank_image[256:, :256] = HL
    blank_image[256:, 256:] = HH
    return blank_image




image = dwt(image)
watermarked = dwt(watermarked)
plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.title('Watermarked : ' + str(wpsnr(watermarked, image)))
plt.imshow(watermarked, cmap='gray')
plt.subplot(133)
diff = np.abs(watermarked - np.float64(image))
diff = np.max(diff) - diff
print(diff)
plt.imshow(diff, cmap='gray')
plt.show()



