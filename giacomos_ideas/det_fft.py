from scipy.fft import fft, ifft
import numpy as np


def embedding_FFT(image, mark_size, alpha, v='multiplicative'):
    # Get the DCT transform of the image
    ori_dct = fft(fft(image, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Get the locations of the most perceptually significant components
    abs_ori_dct = abs(ori_dct)
    locations = np.argsort(-abs_ori_dct, axis=None)  # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]  # locations as (x,y) coordinates

    # Generate a watermark
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    np.save('mark.npy', mark)

    # Embed the watermark
    watermarked_dct = ori_dct.copy()
    for idx, (loc, mark_val) in enumerate(zip(locations[1:], mark)):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= complex(1 + (alpha * mark_val), 0)

    # Restore sign and o back to spatial domain
    # watermarked_dct *= sign
    watermarked = np.uint8(ifft(ifft(watermarked_dct, axis=1, norm='ortho'), axis=0, norm='ortho'))

    return mark, watermarked


def detection_FFT(image, watermarked, alpha, mark_size, v='multiplicative'):
    ori_dct = fft(fft(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    wat_dct = fft(fft(watermarked, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Get the locations of the most perceptually significant components
    abs_ori_dct = abs(ori_dct)
    abs_wat_dct = abs(wat_dct)
    locations = np.argsort(-abs_ori_dct, axis=None)  # - sign is used to get descending order
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

