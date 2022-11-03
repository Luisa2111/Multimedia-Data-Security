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

def embedding_DCT(image, mark, alpha = 0.1, v='multiplicative', plot_mark = False):
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
    mark_places = ori_dct.copy()
    mark_places[:,:] = 1
    for idx, (loc,mark_val) in enumerate(zip(locations[1:], mark)):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + ( alpha * mark_val)
            mark_places[loc] = 0
    # Restore sign and o back to spatial domain
    if plot_mark:
        plt.figure(figsize=(15, 6))
        plt.title('Position for watermark embedding')
        plt.imshow(mark_places, cmap='gray')
        plt.show()
    watermarked_dct *= sign
    watermarked = np.uint8(idct(idct(watermarked_dct,axis=1, norm='ortho'),axis=0, norm='ortho'))
    return watermarked


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


def embedding(name_image, mark, alpha = 10, name_output = 'watermarked.bmp', dim = 8 , step = 15, max_splits = 500):
    # first level
    image = cv2.imread(name_image, 0)

    q = hvs.hvs_step(image, dim = dim, step = step)

    coeffs2 = pywt.dwt2(image, 'haar')
    image, (LH, HL, HH) = coeffs2

    sh = image.shape
    # SUB LEVEL EMBEDDING

    if sh[0] % dim != 0:
        return 'img size not div by ' + str(dim)

    #if mark.size % ((sh[0]//dim)*(sh[1]//dim)) != 0:
    #    return 'mark size not div by ' + str(dim)

    # watermarked = image.copy()
    splits = min(np.count_nonzero(q), max_splits)
    sub_mark = np.array_split(mark, splits)
    sub_mark_size = sub_mark[0].size
    for i in range(len(sub_mark)):
        sub_mark[i] = np.pad(sub_mark[i], (0,sub_mark_size - sub_mark[i].size), 'constant')
    locations = np.argsort(-q,axis=None)
    locations = locations[:splits]
    rows = q.shape[0]
    locations = [(val // rows, val % rows) for val in locations]
    for loc in locations:
        i = loc[0]
        j = loc[1]
        image[i*dim:(i+1)*dim,j*dim:(j+1)*dim] = im_idct(embedding_SVD(im_dct(image[i*dim:(i+1)*dim,j*dim:(j+1)*dim]),
                                                             sub_mark.pop(0), alpha = (q[i, j])*alpha , mode = "d"))



    # second level
    coeffs2_2 = pywt.dwt2(image, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs2_2

    # third level
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

    watermarked =  pywt.idwt2((image, (LH, HL, HH)), 'haar')
    # watermarked =  pywt.idwt2((pywt.idwt2((LL2, (LH2, HL2, HH2)), 'haar'), (LH, HL, HH)), 'haar')
    # return pywt.idwt2((pywt.idwt2((pywt.idwt2((LL3, (LH3, HL3, HH3)), (LH2, HL2, HH2)), 'haar'), (LH, HL, HH)), 'haar')

    cv2.imwrite(name_output, watermarked)
    return watermarked
