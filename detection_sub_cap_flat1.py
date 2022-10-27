from scipy.fft import dct, idct
import numpy as np
from psnr import wpsnr,similarity
import hvs_lambda as hvs
import embedding_flat_file as fl
import cv2, pywt

def im_dct(image):
    return dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')

def im_idct(image):
    return (idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho'))



def extraction_SVD(image, watermarked, alpha, mark_size, mode='additive'):
    """
    Extraction of the mark from the singular values of image
    :param image: original image as nxn numpy matrix
    :param watermarked: watermarked image as nxn numpy matrix
    :param alpha: strength of the embedding
    :param mark_size: expected size of the mark
    :param mode:  multiplicative, additive and the third option is an adaptive version of the additive
    :return: mark
    """
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
    """
    Extraction of the mark. See embedding to have more information
    :param image: original image as nxn numpy matrix
    :param watermarked: watermarked image as nxn numpy matrix
    :param mark_size: size of the mark
    :param alpha: strength of embedding
    :param dim: dimension of blocks in first level DWT
    :param step: quantization steps for hws function
    :param max_splits: max number of blocks to be embedded
    :param min_splits: min number of blocks to be embedded
    :param sub_size: max dimension of submarks
    :param Xi_exp: represent how much we weight Xi from HVS
    :param Lambda_exp: represent how much we weight Lambda from HVS
    :param L_exp: represent how much we weight L from HVS
    :param ceil: if we use ceil or rint when quantizing the HVS
    :return: the extracted mark as numpy array
    """
    # extraction phase
    # first level
    mark = []

    q = hvs.hvs_step(image, dim = dim, step = step, Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp, ceil = ceil)

    # SUB LEVEL EMBEDDING

    # first level
    image, (LH_ori, HL_ori, HH_ori) = pywt.dwt2(image, 'haar')
    watermarked, (LH_wat, HL_wat, HH_wat) = pywt.dwt2(watermarked, 'haar')

    splits = min(np.count_nonzero(q), max_splits)
    locations = np.argsort(-q, axis=None)
    mark_flat = []
    # case of flat images
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
        #extraction from the dark locations
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
    else:
        new_mark_size = mark_size

    if new_mark_size % splits == 0 :
        sub_mark_size = new_mark_size // splits
    else:
        sub_mark_size = new_mark_size // splits + 1
        last_mark_size = sub_mark_size - 1

    # selection of the locations with the embedding
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

def detection(name_original, name_watermarked, name_attacked, mark_size=1024,  threeshold = 2, alpha = 10,
            dim = 8, step = 15, max_splits = 500, min_splits = 170, sub_size = 6
                , Xi_exp = 0.2, Lambda_exp = 0.5, L_exp = 0 , ceil = True):
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

