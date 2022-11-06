import random
import image_processing as ip
from detection_ef26420c import wpsnr
"""
This will just be used for our own testing purposes, not for the challenge. 
We still have to develop an attack strategy for the day of the challenge.
"""


def random_attack(watermarked, output = False):
    i = random.randint(1, 6)
    if i == 1:
        attacked = ip.awgn(watermarked, 5.0, 123)
    elif i == 2:
        attacked = ip.blur(watermarked, [3, 2])
    elif i == 3:
        attacked = ip.sharpening(watermarked, 1, 2)
    elif i == 4:
        attacked = ip.median(watermarked, [3, 5])
    elif i == 5:
        attacked = ip.resizing(watermarked, 0.5)
    elif i == 6:
        attacked = ip.jpeg_compression(watermarked, 75)
    if output:
        # print('Attacked with attack :',i)
        return attacked, i
    # w = ip.wpsnr(original, attacked)
    # print("wPSNR of attacked picture {image_name}: {decibel:.2f}dB".format(image_name=name_image, decibel=w))
    # ip.plotting_images(original, attacked, title=('Attacked {image_name}').format(image_name=name_image))
    else:
        return attacked

def attack_num(watermarked,i, output = False):
    if i == 1:
        attacked = ip.awgn(watermarked, 5.0, 123)
    elif i == 2:
        attacked = ip.blur(watermarked, 1.5)
    elif i == 3:
        attacked = ip.sharpening(watermarked, 1, 1)
    elif i == 4:
        attacked = ip.median(watermarked, [3, 5])
    elif i == 5:
        attacked = ip.resizing(watermarked, 0.5)
    elif i == 6:
        attacked = ip.jpeg_compression(watermarked, 20)
    if output:
        # print('Attacked with attack :',i)
        return attacked, i
    # w = ip.wpsnr(original, attacked)
    # print("wPSNR of attacked picture {image_name}: {decibel:.2f}dB".format(image_name=name_image, decibel=w))
    # ip.plotting_images(original, attacked, title=('Attacked {image_name}').format(image_name=name_image))
    else:
        return attacked

"""
just used and implemented for our own testing purposes
"""

def random_attack_param(image, output = False):
    ori = image.copy()
    w = 0
    while w < 35:
        i = random.randint(1, 6)
        if i == 1:
            attacked = ip.awgn(image, random.uniform(0.5, 10), random.randint(0, 9999))
        elif i == 2:
            attacked = ip.blur(image, random.uniform(0.5, 3))
        elif i == 3:
            attacked = ip.sharpening(image, random.uniform(0.5, 3), random.uniform(0.5, 3))
        elif i == 4:
            attacked = ip.median(image, [random.randint(1, 5) * 2 + 1, random.randint(1, 5) * 2 + 1])
        elif i == 5:
            attacked = ip.resizing(image, random.uniform(0.1, 0.7))
        elif i == 6:
            attacked = ip.jpeg_compression(image, random.randint(1, 75))
        w = wpsnr(ori, image)
    if output:
        # print('Attacked with attack :',i)
        return attacked, i
    # w = ip.wpsnr(original, attacked)
    # print("wPSNR of attacked picture {image_name}: {decibel:.2f}dB".format(image_name=name_image, decibel=w))
    # ip.plotting_images(original, attacked, title=('Attacked {image_name}').format(image_name=name_image))
    else:
        return attacked


def attack_wpsnr_fix(image, i):
    attacked = image.copy()
    w = 0
    while (w < 35 or w > 50) and (i != 3 or w == 0):
        if i == 1:
            attacked = ip.awgn(image, random.uniform(0.5, 10), random.randint(0, 9999))
        elif i == 2:
            attacked = ip.blur(image, random.uniform(0.5, 3))
        elif i == 3:
            attacked = ip.sharpening(image, random.uniform(0.9, 1.1), random.uniform(0.9, 1.1))
        elif i == 4:
            attacked = ip.median(image, [random.randint(1, 5) * 2 + 1, random.randint(1, 5) * 2 + 1])
        elif i == 5:
            attacked = ip.resizing(image, random.uniform(0.1, 0.7))
        elif i == 6:
            attacked = ip.jpeg_compression(image, random.randint(1, 75))
        w = wpsnr(attacked, image)
    return attacked


def combined_attack(original, watermarked, name_image):
    attacked = ip.jpeg_compression(watermarked, 75)
    attacked = ip.awgn(attacked, 5.0, 123)
    attacked = ip.blur(attacked, [3, 2])
    attacked = ip.sharpening(attacked, 1, 1)
    attacked = ip.median(attacked, [3, 5])
    attacked = ip.resizing(attacked, 0.5)

    w = ip.wpsnr(original, attacked)
    print("wPSNR of attacked picture {image_name}: {decibel:.2f}dB".format(image_name=name_image, decibel=w))
    ip.plotting_images(original, attacked, title=('Attacked {image_name}').format(image_name=name_image))
    return attacked

