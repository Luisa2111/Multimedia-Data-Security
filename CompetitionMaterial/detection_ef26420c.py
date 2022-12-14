import cv2
from wpsnr import wpsnr
import numpy as np
from scipy.fft import dct, idct
import pywt

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  # X = np.rint(X)
  # X_star = np.rint(X_star)
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
  return s


def im_dct(image):
    return dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')


def im_idct(image):
    return (idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho'))


def embedding_flat(block, wat):
    block = im_dct(block)
    if not hasattr(wat, "__len__"):
        if wat==1:
            if block[0,1]<=0:
                block[0,1]=1
            if block[1,1]<=0:
                block[1,1]=1
            if block[1,0]<=0:
                block[1,0]=1
        elif wat==-1:
            if block[0,1]>=0:
                block[0,1]=-1
            if block[1,1]>=0:
                block[1,1]=-1
            if block[1,0]>=0:
                block[1,0]=-1
        else:
            print("the wat value must be -1 or 1. Not", wat)
    else:
        print("wat must be an integer not a list")
    return im_idct(block)


def extraction_flat(block):
    block = im_dct(block)
    average=(block[0,1]+block[1,1]+block[1,0])/3
    # if average == 0 : print('average = 0')
    return np.sign(average)


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


def extraction(image, watermarked, mark_size, alpha, dim = 8, step = 15, max_splits = 500, min_splits = 170, sub_size = 6
               , Xi_exp = 0.2, Lambda_exp = 0.5, L_exp = 0, ceil = True):
    # extraction phase
    # first level
    mark = []

    q = hvs_step(image, dim = dim, step = step, Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp, ceil = ceil)

    # SUB LEVEL EMBEDDING

    # first level
    image, (LH_ori, HL_ori, HH_ori) = pywt.dwt2(image, 'haar')
    watermarked, (LH_wat, HL_wat, HH_wat) = pywt.dwt2(watermarked, 'haar')

    splits = min(np.count_nonzero(q), max_splits)
    locations = np.argsort(-q, axis=None)
    mark_flat = []
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
            mark_flat[mark_pos] = extraction_flat(watermarked[i * dim:(i + 1) * dim , j * dim:(j + 1) * dim])
            mark_pos += 1
    else:
        new_mark_size = mark_size

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
    if not mark.any():
        mark[:] = 1
    # print('ex splits', splits, '| submarksize', sub_mark_size, '| flat size', np.count_nonzero(mark_flat))
    return mark


from math import floor
def hvs_quantization_Lambda(image):
    sh = image.shape
    I_30, (I_00, I_20, I_10) = pywt.dwt2(image, 'haar')
    I_31, (I_01, I_21, I_11) = pywt.dwt2(I_30, 'haar')
    I_32, (I_02, I_22, I_12) = pywt.dwt2(I_31, 'haar')
    I_33, (I_03, I_23, I_13) = pywt.dwt2(I_32, 'haar')

    matrix = np.zeros((sh[0] // 2, sh[1] // 2), dtype=np.float64)
    I = [[I_00, I_10, I_20], [I_01, I_11, I_21], [I_02, I_12, I_22], [I_03, I_13, I_23, I_33]]

    for i in range((sh[0] // 2)):
        for j in range((sh[1] // 2)):
            l = 0
            # the eye is less sensitive to noise in those areas of the image where brightness is high or low
            if 1 + floor(i // (2 ** (3 - l))) < 32 and 1 + floor(j // (2 ** (3 - l))) < 32:
                L = 1 / 256 * I[3][3][1 + floor(i // (2 ** (3 - l))), 1 + floor(j // (2 ** (3 - l)))]
                Lambda = 1 + L
            matrix[i, j] = Lambda

    return matrix


def hvs_quantization_L(image):
    sh = image.shape
    I_30, (I_00, I_20, I_10) = pywt.dwt2(image, 'haar')
    I_31, (I_01, I_21, I_11) = pywt.dwt2(I_30, 'haar')
    I_32, (I_02, I_22, I_12) = pywt.dwt2(I_31, 'haar')
    I_33, (I_03, I_23, I_13) = pywt.dwt2(I_32, 'haar')

    matrix = np.zeros((sh[0] // 2, sh[1] // 2), dtype=np.float64)

    I = [[I_00, I_10, I_20], [I_01, I_11, I_21], [I_02, I_12, I_22], [I_03, I_13, I_23, I_33]]

    for i in range((sh[0] // 2)):
        for j in range((sh[1] // 2)):
            l = 0
            # the eye is less sensitive to noise in those areas of the image where brightness is high or low
            if 1 + floor(i // (2 ** (3 - l))) < 32 and 1 + floor(j // (2 ** (3 - l))) < 32:
                L = 1 / 256 * I[3][3][1 + floor(i // (2 ** (3 - l))), 1 + floor(j // (2 ** (3 - l)))]
            if L < 0.5 : L = 1 - L
            matrix[i, j] = L

    return matrix


def hvs_quantization_Xi(image):
    sh = image.shape
    I_30, (I_00, I_20, I_10) = pywt.dwt2(image, 'haar')
    I_31, (I_01, I_21, I_11) = pywt.dwt2(I_30, 'haar')
    I_32, (I_02, I_22, I_12) = pywt.dwt2(I_31, 'haar')
    I_33, (I_03, I_23, I_13) = pywt.dwt2(I_32, 'haar')

    matrix = np.zeros((sh[0] // 2, sh[1] // 2), dtype=np.float64)

    I = [[I_00, I_10, I_20], [I_01, I_11, I_21], [I_02, I_12, I_22], [I_03, I_13, I_23, I_33]]

    for i in range((sh[0] // 2)):
        for j in range((sh[1] // 2)):
            l = 0
            Xi = 0
            small_theta = 2
            # The eye is less sensitive to noise in high resolution bands, and in those bands having orientation of 45
            # the eye is less sensitive to noise in highly textured areas but, among these, more sensitive
            # near the  edges
            for k in range(0, 3 - l):
                for theta in range(0, 2):
                    for x in range(0, 1):
                        for y in range(0, 1):
                            # if y + (i // (2 ** k)) < 64 and x + (j // (2 ** k)) < 64:
                            Xi = Xi + (I[k + l][theta][y + (i // (2 ** k)), x + (j // (2 ** k))]) ** 2
                    Xi = Xi * 1 // (16 ** k)
            Xi = Xi * np.var([[I[3][3][y + i // (2 ** (3 - l)), x + j // (2 ** (3 - l))]
                               for x in [0, 1] if x + j // (2 ** (3 - l)) < 32] for y in [0, 1] if
                              y + i // (2 ** (3 - l)) < 32])

            matrix[i, j] = Xi
    return matrix


def hvs_step(image, dim, step, Xi_exp = 0.2, Lambda_exp = 0.5, L_exp = 0, ceil = True):
    """ dim is the dimension of block in wavelet domain first level, so we need to double it """
    hvs = (hvs_quantization_Xi(image)**Xi_exp)*(hvs_quantization_Lambda(image)**Lambda_exp)*(hvs_quantization_L(image)**L_exp)
    sh = image.shape
    sh_hvs = hvs.shape
    dim_out = (sh[0] // (2 * dim))
    matrix = np.zeros((dim_out, sh[1] // (2 * dim)), dtype=np.float64)
    dim_hvs = sh_hvs[0] // (dim_out)
    for i in range(dim_out):
        for j in range(sh[1] // (2 * dim)):
            matrix[i, j] = np.mean(hvs[i * dim_hvs:(i + 1) * dim_hvs - 1, j * dim_hvs:(j + 1) * dim_hvs - 1])
            if matrix[i,j] != 0:
                # ceil does an over approximation,
                # rint a better approximation, with better robustness
                # but creates false positive
                if ceil:
                    matrix[i,j] = np.ceil(matrix[i,j]/step)
                else:
                    matrix[i, j] = np.rint(matrix[i, j] / step)

    return matrix


def detection(name_original, name_watermarked, name_attacked):
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
    threeshold = 1.62
    mark_size = 1024

    image = cv2.imread(name_original, 0)
    wat_original = cv2.imread(name_watermarked,0)
    wat_attacked = cv2.imread(name_attacked,0)
    extracted_mark = extraction(image, wat_attacked, mark_size, alpha, dim = dim, step = step, max_splits = max_splits, min_splits = min_splits,
                                sub_size = sub_size, Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp , ceil = ceil)
    mark = extraction(image, wat_original,mark_size,alpha, dim = dim, step = step, max_splits = max_splits, min_splits = min_splits,
                                sub_size = sub_size, Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp , ceil = ceil)

    # threeshold and similiarity
    sim = similarity(mark,extracted_mark)
    if sim > threeshold:
        out1 = 1
    else:
        out1 = 0
    return out1, wpsnr(wat_original,wat_attacked)
    