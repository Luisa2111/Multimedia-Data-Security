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

"""

def im_dct(image):
    return dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')

def im_idct(image):
    return (idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho'))


def generate_mark(mark_size):
    # Generate a watermark
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    np.save('mark.npy', mark)
    return mark

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


def embedding_SVD(image,mark, alpha = 1, mode = 'multiplicative'):
    u, s, v = np.linalg.svd(image)
    if mark.size >= s.size :
        print('error mark',mark.size,'diag',s.size)
        return 123

    mark_pad = np.pad(mark , (1, s.size - mark.size - 1) , 'constant')
    if mode == 'multiplicative':
        s *= 1 + alpha * mark_pad
    else :
        s += alpha * mark_pad
    watermarked = np.matrix(u) * np.diag(s) * np.matrix(v)
    return watermarked


import pywt


def embedding(name_image, mark, alpha = 0.1, name_output = 'watermarked.bmp', dim = 16):
    # first level
    image = cv2.imread(name_image, 0)

    q = hvs.hvs_blocks(image, dim = dim)

    coeffs2 = pywt.dwt2(image, 'haar')
    image, (LH, HL, HH) = coeffs2

    sh = image.shape
    # SUB LEVEL EMBEDDING

    if sh[0] % dim != 0:
        return 'img size not div by ' + str(dim)

    #if mark.size % ((sh[0]//dim)*(sh[1]//dim)) != 0:
    #    return 'mark size not div by ' + str(dim)

    # watermarked = image.copy()
    sub_mark = np.array_split(mark, np.count_nonzero(q))
    sub_mark_size = mark.size // np.count_nonzero(q) + 1
    last_mark_size = mark.size % sub_mark_size
    for i in range(len(sub_mark)):
        sub_mark[i] = np.pad(sub_mark[i], (0,sub_mark_size - sub_mark[i].size), 'constant')
    for i in range(0,sh[0],dim):
        for j in range(0,sh[1],dim):
            #if q[i//dim,j//dim] == 0 :
            #   image[i:i + dim - 1, j:j + dim - 1] = embedding_DCT((image[i:i + dim - 1, j:j + dim - 1]),
            #                                                               sub_mark.pop(),alpha=alpha)
            if q[i // dim, j // dim] != 0:
                image[i:i+dim-1,j:j+dim-1] = im_idct(embedding_SVD(im_dct(image[i:i+dim-1,j:j+dim-1]),
                                                             sub_mark.pop(0), alpha = q[i//dim,j//dim] *0.1*alpha))



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