from scipy.fft import dct
import numpy as np


def detection(image, watermarked, alpha, mark_size, v):
    ori_dct = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    wat_dct = dct(dct(watermarked, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Get the locations of the most perceptually significant components
    ori_dct = abs(ori_dct)
    wat_dct = abs(wat_dct)
    locations = np.argsort(-ori_dct, axis=None)  # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]  # locations as (x,y) coordinates

    # Generate a watermark
    w_ex = np.zeros(mark_size, dtype=np.float64)

    # Embed the watermark
    for idx, loc in enumerate(locations[1:mark_size + 1]):
        if v == 'additive':
            w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / alpha
        elif v == 'multiplicative':
            w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / (alpha * ori_dct[loc])

    return w_ex


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


def similarity(x, x_star):
    s = np.sum(np.multiply(x, x_star)) / np.sqrt(np.sum(np.multiply(x_star, x_star)))
    return s


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
