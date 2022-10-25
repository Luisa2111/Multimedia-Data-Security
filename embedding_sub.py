from scipy.fft import dct, idct
import numpy as np
import cv2
import matplotlib.pyplot as plt
import image_processing as ip
import hvs

""" 
NOTES:
version 1 destroied with attack 2,4,5,6

version with svd on the dct domain broken by 1 and 2 (awgn and blur)

in my opinion in texture area we have to do different embedding (these areas are less effected by awgn)

version adaptive additive svd is broken by 2,4,5 but has good quality

"""

def im_dct(image):
    return dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')

def im_idct(image):
    return (idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho'))


# here we will insert differnt possibilities for embedding


def embedding_SVD(image,mark, alpha = 1, mode = 'additive'):
    u, s, v = np.linalg.svd(image)
    if mark.size >= s.size :
        print('error mark',mark.size,'diag',s.size)
        return 123
    if mode == 'multiplicative':
        mark_pad = np.pad(mark, (1, s.size - mark.size - 1), 'constant')
        s *= 1 + alpha * mark_pad
    elif mode == 'additive' :
        mark_pad = np.pad(mark, (1, s.size - mark.size - 1), 'constant')
        s += alpha * mark_pad
    else:
        locations = np.argsort(-s)
        for i in range(len(mark)):
            s[locations[i+1]] += alpha * mark[i]

    watermarked = np.matrix(u) * np.diag(s) * np.matrix(v)
    return watermarked


import pywt


def embedding(name_image, mark, alpha = 10, name_output = 'watermarked.bmp', dim = 8 , step = 15):
    # first level
    image = cv2.imread(name_image, 0)
    q = hvs.hvs_step(image, dim = dim, step = step)

    coeffs2 = pywt.dwt2(image, 'haar')
    image, (LH, HL, HH) = coeffs2

    sh = image.shape
    # SUB LEVEL EMBEDDING

    if sh[0] % dim != 0:
        return 'img size not div by ' + str(dim)

    sub_mark = np.array_split(mark, np.count_nonzero(q))
    sub_mark_size = sub_mark[0].size
    for i in range(len(sub_mark)):
        sub_mark[i] = np.pad(sub_mark[i], (0,sub_mark_size - sub_mark[i].size), 'constant')
    for i in range(0,sh[0],dim):
        for j in range(0,sh[1],dim):
            if q[i // dim, j // dim] != 0:
                image[i:i+dim-1,j:j+dim-1] = im_idct(embedding_SVD(im_dct(image[i:i+dim-1,j:j+dim-1]),
                                                             sub_mark.pop(0), alpha = (q[i // dim, j // dim])*alpha , mode = "d"))

    watermarked = pywt.idwt2((image, (LH, HL, HH)), 'haar')
    cv2.imwrite(name_output, watermarked)
    return watermarked
