from scipy.fft import dct, idct
import numpy as np
import cv2
import image_processing as ip


def generate_mark(mark_size):
    # Generate a watermark
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    np.save('mark.npy', mark)
    return mark

# here we will insert differnt possibilities for embedding

def embedding_DCT(image, mark, alpha = 0.1, v='multiplicative'):
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

import pywt

def embedding(name_image, mark, alpha = 0.1, name_output = 'watermarked.bmp'):
    # first level
    image = cv2.imread(name_image, 0)
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # second level
    coeffs2_2 = pywt.dwt2(LL, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs2_2
    mark1 = mark[:int(mark.size / 2)]
    mark2 = mark[int(mark.size / 2):]
    HL2 = embedding_DCT(HL2, alpha=alpha, mark=mark1)
    LH2 = embedding_DCT(LH2, alpha=alpha, mark=mark2)
    # third level
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

    # return pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    watermarked =  pywt.idwt2((pywt.idwt2((LL2, (LH2, HL2, HH2)), 'haar'), (LH, HL, HH)), 'haar')
    # return pywt.idwt2((pywt.idwt2((pywt.idwt2((LL3, (LH3, HL3, HH3)), (LH2, HL2, HH2)), 'haar'), (LH, HL, HH)), 'haar')

    cv2.imwrite(name_output, watermarked)
    return watermarked
