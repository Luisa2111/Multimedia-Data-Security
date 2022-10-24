from scipy.fft import dct, idct
import numpy as np
from psnr import wpsnr,similarity
import embedding_sub as em

def im_dct(image):
    return dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')

def im_idct(image):
    return (idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho'))


def extraction_DCT(image, watermarked, mark_size, alpha, v='multiplicative'):
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

import cv2, pywt

def extraction_SVD(image, watermarked, alpha, mark_size, mode='multiplicative'):
    u, s, v = np.linalg.svd(image)
    uu, s_wat, vv = np.linalg.svd(watermarked)
    if mode == 'multiplicative':
        w_ex = (s_wat - s)/(alpha*s)
    else:
        w_ex = (s_wat - s)/alpha
    return w_ex[1:mark_size+1]

def extraction(image, watermarked, mark_size, alpha, dim = 32):
    # extraction phase
    # first level
    mark = []
    # SUB LEVEL EMBEDDING

    # first level
    image, (LH_ori, HL_ori, HH_ori) = pywt.dwt2(image, 'haar')
    sh = image.shape
    watermarked, (LH_wat, HL_wat, HH_wat) = pywt.dwt2(watermarked, 'haar')
    sub_mark_size = mark_size // ((sh[0]//dim)*(sh[1]//dim))
    print(sub_mark_size)
    for i in range(0, sh[0], dim):
        for j in range(0, sh[1], dim):
             mark.append((extraction_SVD(im_dct(image[i:i + dim - 1, j:j + dim - 1]), im_dct(watermarked[i:i + dim - 1, j:j + dim - 1])
                                                                      ,mark_size=sub_mark_size,  alpha=alpha)))
    return np.concatenate(mark)

def detection(name_original, name_watermarked, name_attacked, mark,  threeshold = 12, alpha = 0.1,  v = 'multiplicative', ):
    mark_size = mark.size
    image = cv2.imread(name_original, 0)
    wat_original = cv2.imread(name_watermarked,0)
    wat_attacked = cv2.imread(name_attacked,0)
    extracted_mark = extraction(image, wat_attacked, mark_size, alpha)

    # threeshold and similiarity
    sim = similarity(mark,extracted_mark)
    if sim > threeshold:
        out1 = 1
    else:
        out1 = 0
    return out1, wpsnr(wat_original,wat_attacked)


"""
Computes the similarity measure between the original and the new watermarks.
"""

def compute_thr(sim, mark_size, w):
    SIM = np.zeros(1000)
    SIM[0] = abs(sim)
    for i in range(1, 1000):
        r = np.random.uniform(0.0, 1.0, mark_size)
        SIM[i] = abs(similarity(w, r))

    SIM.sort()
    t = SIM[-2]
    T = t + (0.1 * t)
    print(T)
    return T


"""
Computes the similarity measure between the original and the new watermarks.
"""

"""
This function computes the decision whether mark was destroyed or not.
"""


def decision(mark, mark_size, w_ex):
    sim = similarity(mark, w_ex)
    threshold = compute_thr(sim, mark_size, mark)

    if sim > threshold:
        print('Mark has been found. SIM = %f' % sim)
    else:
        print('Mark has been lost. SIM = %f' % sim)
