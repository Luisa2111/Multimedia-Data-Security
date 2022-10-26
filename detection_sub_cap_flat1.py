from scipy.fft import dct, idct
import numpy as np
from psnr import wpsnr,similarity
import hvs
import embedding_flat_file as fl
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
        return w_ex[1:mark_size+1]
    elif mode == 'additive':
        w_ex = (s_wat - s)/alpha
        return w_ex[1:mark_size+1]
    else:
        locations = np.argsort(-s)
        w_ex = np.zeros(mark_size)
        for i in range(mark_size):
            w_ex[i] = (s_wat[locations[i+1]] - s[locations[i+1]])/alpha
        return w_ex

def extraction(image, watermarked, mark_size, alpha, dim = 8, step = 15, max_splits = 500, min_splits = 180, sub_size = 6):
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
    locations = np.argsort(-q, axis=None)
    if splits < min_splits :
        new_mark_size = int(splits * sub_size - 1)
        flat_mark_size = mark_size - new_mark_size
        mark_flat = np.zeros(flat_mark_size)
        dc_coeff = q.copy()
        dc_coeff[:] = 0
        for i in range(dc_coeff.shape[0]):
            for j in range(dc_coeff.shape[1]):
                if q[i,j] != 0:
                    dc_coeff[i,j] = 99999
                else:
                    dct = im_dct(image[i*dim:(i+1)*dim,j*dim:(j+1)*dim])
                    dc_coeff[i, j] = dct[0,0]**0.2 * np.var(np.squeeze(dct)[1:])

        dark_locations = np.argsort(dc_coeff, axis=None)
        dark_locations = dark_locations[:flat_mark_size]
        rows = dc_coeff.shape[0]
        dark_locations = [(val // rows, val % rows) for val in dark_locations]
        mark_pos = 0
        for loc in dark_locations:
            i = loc[0]
            j = loc[1]
            mark_flat[mark_pos] = fl.extraction_flat(watermarked[i * dim:(i + 1) * dim , j * dim:(j + 1) * dim])
            mark_pos += 1

    if new_mark_size % splits == 0 :
        sub_mark_size = new_mark_size // splits
    else:
        sub_mark_size = new_mark_size // splits + 1
        last_mark_size = sub_mark_size - 1

    locations = locations[:splits]
    rows = q.shape[0]
    locations = [(val // rows, val % rows) for val in locations]
    for loc in locations:
        i = loc[0]
        j = loc[1]
        mark.append((extraction_SVD(im_dct(image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim]),
                                    im_dct(watermarked[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim])
                                    , mark_size=sub_mark_size, alpha=(q[i, j]) * alpha, mode="d")))

    if new_mark_size % splits != 0:
        for i in range( new_mark_size % splits, len(mark)):
            mark[i] = mark[i][:last_mark_size]

    # print('num of submarks' ,len(mark))
    mark.append(mark_flat)
    mark = np.concatenate(mark)

    # print('ex splits', splits, '| submarksize', sub_mark_size, '| flat size', np.count_nonzero(mark_flat))
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