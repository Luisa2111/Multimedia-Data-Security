import random
import image_processing as ip

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
        attacked = ip.sharpening(watermarked, 1, 1)
    elif i == 4:
        attacked = ip.median(watermarked, [3, 5])
    elif i == 5:
        attacked = ip.resizing(watermarked, 0.5)
    elif i == 6:
        attacked = ip.jpeg_compression(watermarked, 75)
    if output:
        print('Attacked with attack :',i)
    # w = ip.wpsnr(original, attacked)
    # print("wPSNR of attacked picture {image_name}: {decibel:.2f}dB".format(image_name=name_image, decibel=w))
    # ip.plotting_images(original, attacked, title=('Attacked {image_name}').format(image_name=name_image))
    return attacked


"""
just used and implemented for our own testing purposes
"""


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
