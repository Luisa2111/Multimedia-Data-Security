from scipy.fft import dct, idct
import numpy as np
from psnr import wpsnr,similarity
import hvs
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

def extraction(image, watermarked, mark_size, alpha, dim = 8, step = 15, max_splits = 500):
    # extraction phase
    # first level
    mark = []

    q = hvs.hvs_step(image, dim = dim, step = step)

    # SUB LEVEL EMBEDDING

    # first level
    image, (LH_ori, HL_ori, HH_ori) = pywt.dwt2(image, 'haar')
    sh = image.shape
    watermarked, (LH_wat, HL_wat, HH_wat) = pywt.dwt2(watermarked, 'haar')

    splits = min(np.count_nonzero(q), max_splits)
    if mark_size % splits == 0 :
        sub_mark_size = mark_size // splits
    else:
        sub_mark_size = mark_size // splits + 1
        last_mark_size = sub_mark_size - 1

    locations = np.argsort(-q, axis=None)
    locations = locations[:splits]
    rows = q.shape[0]
    locations = [(val // rows, val % rows) for val in locations]
    for loc in locations:
        i = loc[0]
        j = loc[1]
        mark.append((extraction_SVD(im_dct(image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim]),
                                    im_dct(watermarked[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim])
                                    , mark_size=sub_mark_size, alpha=(q[i, j]) * alpha, mode="d")))


    if mark_size % splits != 0:
        for i in range( mark_size % splits, len(mark)):
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
