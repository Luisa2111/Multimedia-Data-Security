from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_path='lena.bmp'
image = cv2.imread(img_path, 0)

mark = np.random.uniform(0.0, 1.0, 1024)
mark = np.uint8(np.rint(mark))

import pywt
#from google.colab.patches import cv2_imshow

def embedding_DCT(image, mark = mark, alpha = 0.1, v='multiplicative'):
    # Get the DCT transform of the image
    ori_dct = dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')
    # Get the locations of the most perceptually significant components
    sign = np.sign(ori_dct)
    ori_dct = abs(ori_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates
    # Embed the watermark
    watermarked_dct = ori_dct.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations[1:], mark)):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + ( alpha * mark_val)
    # Restore sign and o back to spatial domain
    watermarked_dct *= sign
    watermarked = (idct(idct(watermarked_dct,axis=1, norm='ortho'),axis=0, norm='ortho'))
    return watermarked

#cv2_imshow(embedding_DCT(image, mark = mark))

def detection(image, watermarked, alpha = 0.1, mark_size = mark.size, v='multiplicative'):
    ori_dct = dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')
    wat_dct = dct(dct(watermarked,axis=0, norm='ortho'),axis=1, norm='ortho')
    # Get the locations of the most perceptually significant components
    ori_dct = abs(ori_dct)
    wat_dct = abs(wat_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates
    # Generate a watermark
    w_ex = np.zeros(mark_size, dtype=np.float64)
    # Embed the watermark
    for idx, loc in enumerate(locations[1:mark_size+1]):
        if v=='additive':
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) /alpha
        elif v=='multiplicative':
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) / (alpha*ori_dct[loc])
    return w_ex

def embedding_WDT(image, mark = mark, alpha = 0.1):
  # first level
  coeffs2 = pywt.dwt2(image, 'haar')
  LL, (LH, HL, HH) = coeffs2

  # second level
  coeffs2_2 = pywt.dwt2(LL, 'haar')
  LL2, (LH2, HL2, HH2) = coeffs2_2
  LL2 = embedding_DCT(LL2, alpha = alpha, mark = mark)

  # third level
  LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

  #return pywt.idwt2((LL, (LH, HL, HH)), 'haar')
  return pywt.idwt2((pywt.idwt2((LL2, (LH2, HL2, HH2)), 'haar'), (LH, HL, HH)), 'haar')
  #return pywt.idwt2((pywt.idwt2((pywt.idwt2((LL3, (LH3, HL3, HH3)), (LH2, HL2, HH2)), 'haar'), (LH, HL, HH)), 'haar')

wat_wdt = embedding_WDT(image,alpha = 0.01)

#cv2_imshow(wat_wdt)

wpsnr(wat_wdt,image)
