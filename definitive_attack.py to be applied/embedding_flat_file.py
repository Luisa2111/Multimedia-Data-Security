import numpy as np
from scipy.fft import dct, idct

import matplotlib.pyplot as plt

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

"""
def extraction_flat(block):
    block = im_dct(block)
    average=(block[0,1]+block[1,1]+block[1,0])/3
    # if average == 0 : print('average = 0')
    return np.sign(average)
"""

def embedding_DCT(image, mark, alpha, v='multiplicative', plot_mark = False):
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
    watermarked = (idct(idct(watermarked_dct,axis=1, norm='ortho'),axis=0, norm='ortho'))
    return watermarked

"""
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
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) / alpha
        elif v=='multiplicative':
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) / (alpha*ori_dct[loc])
    return w_ex
"""
