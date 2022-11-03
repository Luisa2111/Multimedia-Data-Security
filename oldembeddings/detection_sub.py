from scipy.fft import dct, idct
import numpy as np
from psnr import wpsnr,similarity
import hvs
import embedding_sub as em

def im_dct(image):
    return dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')

def im_idct(image):
    return (idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho'))



import cv2, pywt

def extraction_SVD(image, watermarked, alpha, mark_size, mode='additive'):
    u, s, v = np.linalg.svd(image)
    uu, s_wat, vv = np.linalg.svd(watermarked)
    if mode == 'multiplicative':
        w_ex = (s_wat - s)/(alpha*s)
        return w_ex[1:mark_size + 1]
    elif mode == 'additive':
        w_ex = (s_wat - s)/alpha
        return w_ex[1:mark_size+1]
    else:
        locations = np.argsort(-s)
        w_ex = np.zeros(mark_size)
        for i in range(mark_size):
            w_ex[i] = (s_wat[locations[i+1]] - s[locations[i+1]])/alpha
        return w_ex

def extraction(image, watermarked, mark_size, alpha, dim = 8, step = 15):
    # extraction phase
    # first level
    mark = []

    q = hvs.hvs_step(image, dim = dim, step = step)

    # SUB LEVEL EMBEDDING

    # first level
    image, (LH_ori, HL_ori, HH_ori) = pywt.dwt2(image, 'haar')
    sh = image.shape
    watermarked, (LH_wat, HL_wat, HH_wat) = pywt.dwt2(watermarked, 'haar')

    if mark_size % np.count_nonzero(q) == 0 :
        sub_mark_size = mark_size // np.count_nonzero(q)
    else:
        sub_mark_size = mark_size // np.count_nonzero(q) + 1
        last_mark_size = sub_mark_size - 1

    for i in range(0, sh[0], dim):
        for j in range(0, sh[1], dim):
            if q[i // dim, j // dim] != 0:
                mark.append((extraction_SVD(im_dct(image[i:i + dim - 1, j:j + dim - 1]), im_dct(watermarked[i:i + dim - 1, j:j + dim - 1])
                                                                      ,mark_size=sub_mark_size,  alpha=(q[i // dim, j // dim]) * alpha, mode = "d")))

    if mark_size % np.count_nonzero(q) != 0:
        for i in range( mark_size % np.count_nonzero(q), len(mark)):
            mark[i] = mark[i][:last_mark_size]

    mark = np.concatenate(mark)
    return mark

def detection(name_original, name_watermarked, name_attacked, mark_size=1024,  threeshold = 2, alpha = 10):
    image = cv2.imread(name_original, 0)
    wat_original = cv2.imread(name_watermarked,0)
    wat_attacked = cv2.imread(name_attacked,0)
    extracted_mark = extraction(image, wat_attacked, mark_size, alpha)
    mark = extraction(image, wat_original,mark_size,alpha)
    # threeshold and similiarity
    sim = similarity(mark,extracted_mark)
    if sim > threeshold:
        out1 = 1
    else:
        out1 = 0
    return out1, wpsnr(wat_original,wat_attacked)
