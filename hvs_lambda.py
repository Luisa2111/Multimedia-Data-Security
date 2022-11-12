from math import sqrt, floor
import numpy as np
from statistics import variance
import pywt

import matplotlib.pyplot as plt

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


def hvs_blocks(image, dim = 16, Xi_exp = 0.2, Lambda_exp = 0.5, L_exp = 0):
    """ dim is the dimension of block in wavelet domain first level, so we need to double it """
    hvs = (hvs_quantization_Xi(image)**Xi_exp)*(hvs_quantization_Lambda(image)**Lambda_exp)*(hvs_quantization_L(image)**L_exp)
    sh = image.shape
    sh_hvs = hvs.shape
    dim_out = (sh[0] // (2*dim))
    matrix = np.zeros((dim_out, sh[1] // (2*dim)), dtype=np.float64)
    dim_hvs = sh_hvs[0] // (dim_out)
    for i in range(dim_out):
        for j in range(sh[1] // (2*dim)):
            matrix[i,j] = np.mean(hvs[i*dim_hvs:(i+1)*dim_hvs-1,j*dim_hvs:(j+1)*dim_hvs-1])
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


if __name__ == "__main__":
    import cv2
    image = cv2.imread('lena.bmp', 0)
    plt.figure(figsize=(15, 6))
    plt.subplot(141)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.subplot(142)
    plt.title('Matrix of Xi values')
    matrix = hvs_quantization_Xi(image)**0.2
    plt.imshow(matrix, cmap='gray')
    plt.subplot(143)
    matrix = hvs_quantization_Lambda(image) ** 0.3
    plt.title('Matrix of Lambda values')
    plt.imshow(matrix, cmap='gray')
    plt.subplot(144)
    hvs = hvs_step(image, dim=8, step=20, Xi_exp=0.2, Lambda_exp=0.3, L_exp=0.2)
    plt.title('Blocks, non zero = ' + str(np.count_nonzero(hvs)))
    print('max', np.max(hvs), '| != 0', np.count_nonzero(hvs))
    plt.imshow(hvs, cmap='gray')


    plt.show()

