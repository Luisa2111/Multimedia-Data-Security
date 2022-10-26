import numpy as np
from scipy.fft import dct, idct

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