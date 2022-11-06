import cv2
import image_processing as ip
import psnr as psnr
import numpy as np
import embedding_ef26420c as emb
import detection_ef26420c as det
import bf_attack as bfa



def awgn_blurring(name_originalImage, name_watermarkedImage):
    #settings
    sigma_min_blur = 1
    sigma_max_blur = 50
    sigma_step_blur = 1
    std_min_awgn = 0
    std_max_awgn = 80
    std_step_awgn = 1
    seed_awgn = 123
    direction = 1
    attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma = awgn_blurring_bf(name_originalImage, name_watermarkedImage, sigma_min_blur, sigma_max_blur, sigma_step_blur, std_min_awgn, std_max_awgn, std_step_awgn, seed_awgn, direction)
    if hasattr(attackedImage, "__len__"):
        # I'm sure that with std-1 the watermark will be present as I want
        attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma = awgn_blurring_bf(name_originalImage, name_watermarkedImage,sigma -1, sigma, 0.1 ,std - 1, std, 0.1, 123, -1)
        if hasattr(attackedImage, "__len__"):
            # I'm sure that with std+0.1 the watermark will be NOT present as I want
            attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma = awgn_blurring_bf(name_originalImage,name_watermarkedImage,sigma, sigma +0.1, 0.01, std, std + 0.1,0.01, 123, 1)
            if hasattr(attackedImage, "__len__"):
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma
    return 0, "awgn-blurring does not work", "awgn-blurring does not work", "awgn-blurring does not work", "awgn-blurring does not work"

def awgn_blurring_bf(name_originalImage, name_watermarkedImage, sigma_min_blur, sigma_max_blur, sigma_step_blur, std_min_awgn, std_max_awgn, std_step_awgn, seed_awgn, direction):
    originalImage = cv2.imread(name_originalImage, 0)
    watermarkedImage = cv2.imread(name_watermarkedImage, 0)
    name_attackedImage = './attacked/attacked_awgn_blurring.bmp'
    if direction == 1:
        # go this way->
        std = std_min_awgn
        sigma = sigma_min_blur
        decisionMade = 1
        while std < std_max_awgn and sigma < sigma_max_blur and decisionMade == 1:
            attackedImage = ip.awgn(watermarkedImage, std, seed_awgn)
            attackedImage = ip.blur(attackedImage, sigma)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage,name_attackedImage)
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
            decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage,
                                                                 name_attackedImage)
            if decisionMade == 1:
                decisionMade = 1
                return attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma
            sigma = sigma - sigma_step_blur
            std = std - std_step_awgn
    return 0, "awgn-blurring does not work", "awgn-blurring does not work", "awgn-blurring does not work","awgn-blurring does not work"

def awgn_blurring_optimized(name_originalImage, name_watermarkedImage, std, sigma, wpsnr):
    print("2. Wir sind hier")
    std_min = 19
    sigma_min = 0
    seed_awgn = 123
    originalImage = cv2.imread(name_originalImage, 0)
    watermarkedImage = cv2.imread(name_watermarkedImage, 0)
    name_attackedImage = './attacked/attacked_awgn_blurring_optimized.bmp'
    while std_min < std:
        print("std_min")
        while sigma_min < sigma:
            attackedImage = ip.awgn(watermarkedImage, std, seed_awgn)
            attackedImage = ip.blur(attackedImage, sigma)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage,name_attackedImage)
            if decisionMade == 0 and wpsnrWatermarkAttacked > wpsnr:
                return attackedImage,wpsnrWatermarkAttacked, decisionMade, std, sigma
            sigma_min += 0.01
        std_min += 1
        sigma_min = 0
    return 0, "awgn-blurring does not work", "awgn-blurring does not work", "awgn-blurring does not work", "awgn-blurring does not work"


def awgn_blurring_optimized_2(name_originalImage, name_watermarkedImage, std, sigma, wpsnr):
    std_min = 19
    sigma_min = 0
    seed_awgn = 123
    originalImage = cv2.imread(name_originalImage, 0)
    watermarkedImage = cv2.imread(name_watermarkedImage, 0)
    name_attackedImage = './attacked/attacked_awgn_blurring_optimized.bmp'
    std_best = 0
    sigma_best = 0
    wpsnr_best = 0
    decisionMade = 1
    while std_min < std:
        print("1. while Schleife")
        sigma_min = 0
        while sigma_min < sigma and decisionMade == 1:
            attackedImage = ip.awgn(watermarkedImage, std_min, seed_awgn)
            attackedImage = ip.blur(attackedImage, sigma_min)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage,name_attackedImage)
            sigma_min += 0.01
            if decisionMade == 0 and wpsnrWatermarkAttacked > wpsnr_best:
                wpsnr_best = wpsnrWatermarkAttacked
                sigma_best = sigma_min
                std_best = std_min
            std_min += 1
        while sigma_min < sigma:
            print("2. while Schleife")
            attackedImage = ip.awgn(watermarkedImage, std_min, seed_awgn)
            attackedImage = ip.blur(attackedImage, sigma_min)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage,name_attackedImage)
            sigma_min += 0.01
            if decisionMade == 0 and wpsnrWatermarkAttacked > wpsnr_best:
                wpsnr_best = wpsnrWatermarkAttacked
                sigma_best = sigma_min
                std_best = std_min
            std_min += 0.1
        print(wpsnr_best)



        #std_min -= 0.1
        #sigma_min = 0
    """
    while std_min < std:
        print("3. while Schleife")
        while sigma_min < sigma:
            attackedImage = ip.awgn(watermarkedImage, std_min, seed_awgn)
            attackedImage = ip.blur(attackedImage, sigma_min)
            cv2.imwrite(name_attackedImage, attackedImage)
            decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage,name_attackedImage)
            sigma_min += 0.01
            if decisionMade == 0 and wpsnrWatermarkAttacked > wpsnr:
                wpsnr_best = wpsnrWatermarkAttacked
                sigma_best = sigma_min
                std_best = std_min
        std_min += 0.01
        sigma_min = 0
    """
    return attackedImage, wpsnr_best, decisionMade, std_best, sigma_best


"""
def median_blurring(name_originalImage, name_watermarkedImage):
def resizing_blurring(name_originalImage, name_watermarkedImage):
"""









name_mark="./ef26420c.npy"
#name_originalImage = "../sample-images-roc/{num:04d}.bmp".format(num=string_image)
string_image = input("Enter name of the image without '.bmp': ")

name_originalImage = "./sample-images-roc/"+string_image+".bmp"
print(name_originalImage)
cv2.imread("./sample_images_roc/"+string_image+".bmp", 0)
name_watermarkedImage = "./embedded_images/" + string_image + ".bmp"

attackedImage, wpsnrWatermarkAttacked, decisionMade, std, sigma = awgn_blurring(name_originalImage, name_watermarkedImage)
print("std: " + str(std) + ", sigma: " + str(sigma) + " , wpsnr: " + str(wpsnrWatermarkAttacked))
# new brute-force
#attackedImage, wpsnrWatermarkAttacked, decisionMade, std_awgn_single = bfa.awgn_bf_best(name_originalImage, name_watermarkedImage)
#attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma_blur_single = bfa.blur_bf_best(name_originalImage, name_watermarkedImage)
#attackedImage, wpsnr, decisionMade, std, sigma = awgn_blurring_optimized_2(name_originalImage, name_watermarkedImage, std_awgn_single, sigma_blur_single, wpsnrWatermarkAttacked)
#print("std: " + str(std) + ", sigma: " + str(sigma) + " , wpsnr: " + str(wpsnr))
#median_blurring(name_originalImage, name_watermarkedImage)
#resizing_blurring(name_originalImage, name_watermarkedImage)
