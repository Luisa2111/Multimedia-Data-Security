import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from math import sqrt
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

"""
AWGN (Additive White Gaussian Noise) is used to add a normally distributed noise to an image.
"""


def awgn(img, std, seed):
    mean = 0.0  # some constant
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked


"""
Blurring: The blurring is used to remove "outlier" pixels that might be caused by noise. This filter uses a kernel 
of pixels that, sliding through the whole image, changes the value of the pixel in the middle of the kernel with a 
weighted average of the values of the pixels in the kernel. In a Gaussian blur, the pixels that are closer to the 
center of the kernel are given more weight than those far away from the center. The standard deviations (sigma) of 
the Gaussian filter, which affects the size of the kernel, can be an integer or a list of two integers.
"""


def blur(img, sigma):
    from scipy.ndimage.filters import gaussian_filter
    attacked = gaussian_filter(img, sigma)
    return attacked


"""
The sharpening filter is the reverse of blurring. It is used to enhance details in an image by exaggerating the 
brightness difference along edges within an image.
"""


def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)

    attacked = img + alpha * (img - filter_blurred_f)
    return attacked


"""
The median filter is a digital filtering technique often used to remove noise from a signal or an image. The 
filter uses a sliding window that replaces the pixel value at the center of the window with the median of all the 
pixels in the window. The kernel_size can be a scalar or an 2-length list giving the size of the median filter window 
in each dimension. Elements of kernel_size should be odd. If kernel_size is a scalar, then this scalar is used as the 
size in each dimension.
"""


def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked


"""
Resizing is a common operation that can be performed on images.
When an image is downscaled, some information will be lost as the final image will contain less pixels.
When upscaled the values of the new pixels are estimated using interpolation.
"""


def resizing(img, scale):
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1 / scale)
    attacked = attacked[:x, :y]
    return attacked


"""
Images require a lot of bandwidth and storage capacity. Compression is aimed at reducing the amount of data to be 
transmitted. One of the main standards is JPEG. The JPEG is a lossy scheme, i.e. some information is lost during the 
process. By increasing the compression rate, artifacts appear: blocking, blurring, chromatic aberrations. In Python 
we can specify the Quality Factor (QF ∈ [0, 100]$). They lower the QF the higher the compression rate.
"""


def jpeg_compression(img, qf):
    img = Image.fromarray(img)
    img.save('tmp.jpg', "JPEG", quality=qf)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype=np.uint8)
    # os.remove('tmp.jpg')
    return attacked


"""
This function is used to calculate the wpsnr of an watermarked and/or attacked image
"""


def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0

    difference = img1 - img2
    same = not np.any(difference)
    if same is True:
        return 9999999
    csf = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels


"""
This function is used to plot two images.
"""


def plotting_images(img1, img2, title):
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img1, cmap='gray')
    plt.subplot(122)
    plt.title(str(title))
    plt.imshow(img2, cmap='gray')
    plt.show()
