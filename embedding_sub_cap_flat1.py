from scipy.fft import dct, idct
import numpy as np
import image_processing as ip
import hvs
import embedding_flat_file as fl
import attack as at
from psnr import similarity, wpsnr
import cv2
import matplotlib.pyplot as plt

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


def embedding(name_image, mark, alpha = 10, name_output = 'watermarked.bmp', dim = 8 , step = 15, max_splits = 500, min_splits = 180, sub_size = 6):
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
    locations = np.argsort(-q, axis=None)
    if splits < min_splits :
        new_mark_size = int(splits * sub_size - 1)
        flat_mark_size = mark.size - new_mark_size
        mark_flat = mark[new_mark_size:]
        mark = mark[:new_mark_size]
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
        """plt.figure(figsize=(15, 6))
        plt.subplot(131)
        plt.title('Original')
        plt.imshow(image, cmap='gray')
        plt.subplot(132)
        plt.title('Texture')
        plt.imshow(q, cmap='gray')
        plt.subplot(133)
        plt.title('Darkness')
        plt.imshow(dc_coeff, cmap='gray')
        plt.draw()"""
        for loc in dark_locations:
            i = loc[0]
            j = loc[1]
            image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim ] = fl.embedding_flat(image[i * dim:(i + 1) * dim , j * dim:(j + 1) * dim],
                                                                              wat = mark_flat[mark_pos])
            # image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = fl.embedding_DCT(
            #     image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim],
            #     mark = np.repeat(mark_flat[mark_pos], 5), alpha= 0.1
            # )
            mark_pos += 1



    locations = locations[:splits]
    rows = q.shape[0]
    locations = [(val // rows, val % rows) for val in locations]

    sub_mark = np.array_split(mark, splits)
    # print('num of submarks', len(sub_mark))
    sub_mark_size = sub_mark[0].size
    for i in range(len(sub_mark)):
        sub_mark[i] = np.pad(sub_mark[i], (0,sub_mark_size - sub_mark[i].size), 'constant')

    for loc in locations:
        i = loc[0]
        j = loc[1]
        image[i*dim:(i+1)*dim,j*dim:(j+1)*dim] = im_idct(embedding_SVD(im_dct(image[i*dim:(i+1)*dim,j*dim:(j+1)*dim]),
                                                             sub_mark.pop(0), alpha = (q[i, j])*alpha , mode = "d"))


    # print('em splits', splits, '| submarksize', sub_mark_size, '| flat size',flat_mark_size)
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


if __name__ == "__main__":
    alpha = 10
    dim = 8
    step = 13
    max_splits = 500
    # generate a watermark (in the challenge, we will be provided a mark)
    MARK = np.load('ef26420c.npy')
    mark = np.array([(-1) ** m for m in MARK])

    # embed watermark into three different pictures of 512x512 (as it will be in the challenge)
    pictures = ['watermarking-images/lena.bmp', 'watermarking-images/baboon.bmp', 'watermarking-images/cameraman.tif']
    name_image = 'sample-images-roc/0031.bmp'
    # name_image = 'lena.bmp'
    image = cv2.imread(name_image, 0)
    name_out = 'wat_0031.bmp'
    # name_out = 'wat_lena.bmp'
    embedding(name_image, mark, alpha, name_output=name_out, dim=dim, step=step, max_splits=max_splits)
    watermarked = cv2.imread(name_out, 0)
    atk = at.attack_num(watermarked, 2)
    name_atk = 'atk_0031.bmp'
    cv2.imwrite(name_atk, atk)
    # print(dt.detection(name_image,name_out,name_atk,mark,T,alpha))
    plt.figure(figsize=(15, 6))
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.title('Watermarked : ' + str(wpsnr(watermarked, image)))
    plt.imshow(watermarked, cmap='gray')
    plt.subplot(133)
    plt.title('Attacked : ' + str(wpsnr(watermarked, atk)))
    plt.imshow(atk, cmap='gray')
    plt.show()