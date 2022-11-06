import cv2
import image_processing as ip
import psnr as psnr
import numpy as np
import embedding_ef26420c as emb
import detection_ef26420c as det
import bf_attack as bfa
import detection_caller as det_c
import os



def awgn_blurring(name_originalImage, name_watermarkedImage, name_attackedImage):
    #settings
    sigma_min_blur = 1
    sigma_max_blur = 50
    sigma_step_blur = 1
    std_min_awgn = 0
    std_max_awgn = 80
    std_step_awgn = 1
    seed_awgn = 123
    direction = 1
    attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma = awgn_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage, sigma_min_blur, sigma_max_blur, sigma_step_blur, std_min_awgn, std_max_awgn, std_step_awgn, seed_awgn, direction)
    if hasattr(attackedImage, "__len__"):
        # I'm sure that with std-1 the watermark will be present as I want
        attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma = awgn_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage,sigma -1, sigma, 0.1 ,std - 1, std, 0.1, 123, -1)
        if hasattr(attackedImage, "__len__"):
            # I'm sure that with std+0.1 the watermark will be NOT present as I want
            attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma = awgn_blurring_bf(name_originalImage,name_watermarkedImage,name_attackedImage,sigma, sigma +0.1, 0.01, std, std + 0.1,0.01, 123, 1)
            if hasattr(attackedImage, "__len__"):
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma
    return 0, "awgn-blurring does not work", "awgn-blurring does not work", "awgn-blurring does not work", "awgn-blurring does not work"

def awgn_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage, sigma_min_blur, sigma_max_blur, sigma_step_blur, std_min_awgn, std_max_awgn, std_step_awgn, seed_awgn, direction):
    originalImage = cv2.imread(name_originalImage, 0)
    watermarkedImage = cv2.imread(name_watermarkedImage, 0)
    name_attackedImage=name_attackedImage[:-4]+'_awgn_blurring.bmp'
    if direction == 1:
        # go this way->
        std = std_min_awgn
        sigma = sigma_min_blur
        decisionMade = 1
        while std < std_max_awgn and sigma < sigma_max_blur and decisionMade == 1:
            attackedImage = ip.awgn(watermarkedImage, std, seed_awgn)
            attackedImage = ip.blur(attackedImage, sigma)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage,name_attackedImage)
            if decisionMade == 0:
                decisionMade = 0
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma
            if wpsnrWatermarkAttacked < 35:
                print("wpsnrWatermarkAttacked is less than 35...")
                break
            sigma = sigma + sigma_step_blur
            std = std + std_step_awgn
    elif direction == -1:
        # go the other way <-
        std = std_max_awgn
        sigma = sigma_max_blur
        decisionMade = 0
        while std > std_min_awgn and sigma > sigma_min_blur and decisionMade == 0:
            attackedImage = ip.awgn(watermarkedImage, std, seed_awgn)
            attackedImage = ip.blur(attackedImage, sigma)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage,
                                                                 name_attackedImage)
            if decisionMade == 1:
                decisionMade = 1
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma
            sigma = sigma - sigma_step_blur
            std = std - std_step_awgn
    return 0, "awgn-blurring does not work", "awgn-blurring does not work", "awgn-blurring does not work","awgn-blurring does not work"

def median_blurring(name_originalImage, name_watermarkedImage, name_attackedImage):
    #settings
    sigma_min_blur = 1
    sigma_max_blur = 50
    sigma_step_blur = 1
    scale_min_median = 3
    scale_max_median = 9
    scale_step_median = 2
    direction = 1
    attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size, sigma = median_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage, sigma_min_blur, sigma_max_blur, sigma_step_blur, scale_min_median, scale_max_median, scale_step_median, direction)
    if hasattr(attackedImage, "__len__"):
        attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size, sigma = median_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage,sigma -1, sigma, 0.1 ,scale_min_median, scale_max_median, scale_step_median, -1)
        if hasattr(attackedImage, "__len__"):
            attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size, sigma = median_blurring_bf(name_originalImage,name_watermarkedImage,name_attackedImage,sigma, sigma +0.1, 0.01,scale_min_median, scale_max_median, scale_step_median, 1)
            if hasattr(attackedImage, "__len__"):
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size, sigma
    return 0, "median-blurring does not work", "median-blurring does not work", "median-blurring does not work", "median-blurring does not work"

def median_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage, sigma_min_blur, sigma_max_blur, sigma_step_blur, scale_min_median, scale_max_median, scale_step_median, direction):
    originalImage = cv2.imread(name_originalImage, 0)
    watermarkedImage = cv2.imread(name_watermarkedImage, 0)
    name_attackedImage=name_attackedImage[:-4]+'_median_blurring.bmp'
    if direction == 1:
        # go this way->
        scale = scale_min_median
        sigma = sigma_min_blur
        decisionMade = 1
        while scale <= scale_max_median and sigma < sigma_max_blur and decisionMade == 1:
            attackedImage = ip.median(watermarkedImage, scale)
            attackedImage = ip.blur(attackedImage, sigma)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage,name_attackedImage)
            if decisionMade == 0:
                decisionMade = 0
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale, sigma
            if wpsnrWatermarkAttacked < 35:
                print("wpsnrWatermarkAttacked is less than 35...")
                break
            sigma = sigma + sigma_step_blur
            scale = scale + scale_step_median
    elif direction == -1:
        # go the other way <-
        scale = scale_max_median
        sigma = sigma_max_blur
        decisionMade = 0
        while scale >= scale_min_median and sigma > sigma_min_blur and decisionMade == 0:
            attackedImage = ip.blur(watermarkedImage, sigma)
            attackedImage = ip.median(attackedImage, scale)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage,
                                                                 name_attackedImage)
            if decisionMade == 1:
                decisionMade = 1
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale, sigma

            sigma = sigma - sigma_step_blur
            scale = scale - scale_step_median
    return 0, "median-blurring does not work", "median-blurring does not work", "median-blurring does not work","median-blurring does not work"

def resizing_blurring(name_originalImage, name_watermarkedImage, name_attackedImage):
    #settings
    sigma_min_blur = 1
    sigma_max_blur = 50
    sigma_step_blur = 1
    scale_min_resize = 0
    scale_max_resize = 4
    scale_step_resize = 2
    direction = 1
    attackedImage, wpsnrWatermarkAttacked, decisionMade, scale, sigma = resizing_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage, sigma_min_blur, sigma_max_blur, sigma_step_blur, scale_min_resize, scale_max_resize, scale_step_resize, direction)
    if hasattr(attackedImage, "__len__"):
        attackedImage, wpsnrWatermarkAttacked, decisionMade, scale, sigma = resizing_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage,sigma -1, sigma, 0.1,scale, scale+1, 0.1 , -1)
        if hasattr(attackedImage, "__len__"):
            attackedImage, wpsnrWatermarkAttacked, decisionMade, scale, sigma = resizing_blurring_bf(name_originalImage,name_watermarkedImage,name_attackedImage,sigma, sigma +0.1, 0.01,scale-0.1, scale, 0.01, 1)
            if hasattr(attackedImage, "__len__"):
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale, sigma
    return 0, "resize-blurring does not work", "resize-blurring does not work", "resize-blurring does not work", "resize-blurring does not work"

def resizing_blurring_bf(name_originalImage, name_watermarkedImage,name_attackedImage, sigma_min_blur, sigma_max_blur, sigma_step_blur, scale_min_resize, scale_max_resize, scale_step_resize, direction):
    originalImage = cv2.imread(name_originalImage, 0)
    watermarkedImage = cv2.imread(name_watermarkedImage, 0)
    name_attackedImage=name_attackedImage[:-4]+'_resizing_blurring.bmp'
    if direction == 1:
        # go this way->
        scale = scale_max_resize
        sigma = sigma_min_blur
        decisionMade = 1
        while scale >= scale_min_resize and sigma < sigma_max_blur and decisionMade == 1:
            attackedImage = ip.resizing(watermarkedImage, scale)
            attackedImage = ip.blur(attackedImage, sigma)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage,name_attackedImage)
            if decisionMade == 0:
                decisionMade = 0
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale, sigma
            if wpsnrWatermarkAttacked < 35:
                print("wpsnrWatermarkAttacked is less than 35...")
                break
            sigma = sigma + sigma_step_blur
            scale = scale - scale_step_resize
    elif direction == -1:
        # go the other way <-
        scale = scale_min_resize
        sigma = sigma_max_blur
        decisionMade = 0
        while scale <= scale_max_resize and sigma > sigma_min_blur and decisionMade == 0:
            attackedImage = ip.blur(watermarkedImage, sigma)
            attackedImage = ip.resizing(attackedImage, scale)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage,
                                                                 name_attackedImage)
            if decisionMade == 1:
                decisionMade = 1
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale, sigma

            sigma = sigma - sigma_step_blur
            scale = scale + scale_step_resize
    return 0, "resize-blurring does not work", "resize-blurring does not work", "resize-blurring does not work","resize-blurring does not work"


def awgn_median(name_originalImage, name_watermarkedImage, name_attackedImage):
    #settings
    scale_min_median = 3
    scale_max_median = 9
    scale_step_median = 2
    std_min_awgn = 0
    std_max_awgn = 80
    std_step_awgn = 1
    seed_awgn = 123
    direction = 1
    attackedImage, wpsnrWatermarkAttacked, decisionMade, std, scale = awgn_median_bf(name_originalImage, name_watermarkedImage,name_attackedImage, scale_min_median, scale_max_median, scale_step_median, std_min_awgn, std_max_awgn, std_step_awgn, seed_awgn, direction)
    if hasattr(attackedImage, "__len__"):
        # I'm sure that with std-1 the watermark will be present as I want
        attackedImage, wpsnrWatermarkAttacked, decisionMade, std, scale = awgn_median_bf(name_originalImage, name_watermarkedImage,name_attackedImage,scale_min_median, scale_max_median, scale_step_median ,std - 1, std, 0.1, 123, -1)
        if hasattr(attackedImage, "__len__"):
            # I'm sure that with std+0.1 the watermark will be NOT present as I want
            attackedImage, wpsnrWatermarkAttacked, decisionMade, std, scale = awgn_median_bf(name_originalImage,name_watermarkedImage,name_attackedImage,scale_min_median, scale_max_median, scale_step_median, std, std + 0.1,0.01, 123, 1)
            if hasattr(attackedImage, "__len__"):
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, std, scale
    return 0, "awgn-median does not work", "awgn-median does not work", "awgn-median does not work", "awgn-median does not work"

def awgn_median_bf(name_originalImage, name_watermarkedImage,name_attackedImage, scale_min_median, scale_max_median, scale_step_median, std_min_awgn, std_max_awgn, std_step_awgn, seed_awgn, direction):
    originalImage = cv2.imread(name_originalImage, 0)
    watermarkedImage = cv2.imread(name_watermarkedImage, 0)
    name_attackedImage=name_attackedImage[:-4]+'_awgn_median.bmp'
    if direction == 1:
        # go this way->
        std = std_min_awgn
        scale = scale_min_median
        decisionMade = 1
        while std < std_max_awgn and scale <= scale_max_median and decisionMade == 1:
            attackedImage = ip.awgn(watermarkedImage, std, seed_awgn)
            attackedImage = ip.median(attackedImage, scale)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage,name_attackedImage)
            if decisionMade == 0:
                decisionMade = 0
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, std, scale
            if wpsnrWatermarkAttacked < 35:
                print("wpsnrWatermarkAttacked is less than 35...")
                break
            scale = scale + scale_step_median
            std = std + std_step_awgn
    elif direction == -1:
        # go the other way <-
        std = std_max_awgn
        scale = scale_max_median
        decisionMade = 0
        while std > std_min_awgn and scale >= scale_min_median and decisionMade == 0:
            attackedImage = ip.awgn(watermarkedImage, std, seed_awgn)
            attackedImage = ip.median(attackedImage, scale)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage,
                                                                 name_attackedImage)
            if decisionMade == 1:
                decisionMade = 1
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, std, scale
            scale = scale - scale_step_median
            std = std - std_step_awgn
    return 0, "awgn-median does not work", "awgn-median does not work", "awgn-median does not work","awgn-median does not work"

"""
name_mark="./ef26420c.npy"
#name_originalImage = "../sample-images-roc/{num:04d}.bmp".format(num=string_image)
name_images_competitors=os.listdir("./images_of_competition")


for name_image_to_be_attacked in name_images_competitors:
    print("I'm trying to attack " + str(name_image_to_be_attacked))
    name_watermarkedImage = "./images_of_competition/" + str(name_image_to_be_attacked)
    name_originalImage = "./sample_images_roc/" + str(name_image_to_be_attacked[-8:])
    name_attackedImage = "./attacked/" + str(name_image_to_be_attacked[:-9]) + "/ef26420c_" + str(name_image_to_be_attacked)

    #attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma = awgn_blurring(name_originalImage, name_watermarkedImage, name_attackedImage)
    #print("std: " + str(std) + ", sigma: " + str(sigma) + " , wpsnr: " + str(wpsnrWatermarkAttacked))
    attackedImage, wpsnrWatermarkAttacked, decisionMade, qf, sigma = jpeg_blurring(name_originalImage, name_watermarkedImage, name_attackedImage)
    print("qf: " + str(qf) + ", sigma: " + str(sigma) + " , wpsnr: " + str(wpsnrWatermarkAttacked))
"""