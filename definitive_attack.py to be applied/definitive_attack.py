import os
import time

import bf_attack as bf
import localised_attack as la
import bf_composition_attacks as bfa
import image_processing as ip
import cv2
def printer_of_best(nameOfAttack, nameOfvariable, valueOfvariable, wpsnr, attackedImage):
	if hasattr(attackedImage, "__len__"):
		print("BEST ", nameOfAttack)
		print(nameOfAttack, "_", nameOfvariable, "=", valueOfvariable)
		print(nameOfAttack, "_wpsnrWatermarkAttacked=", wpsnr)
	else:
		print(nameOfAttack, " does not work")


def definitive_attack(name_originalImage, name_watermarkedImage, name_attackedImage):
	#this function try first all the attacks alone
	#then try to localise them
	#then try to combine them
	print("Against", name_watermarkedImage)
	#it should be returned the best wpsnr

	print("BRUTE FORCE")
	print("AWGN: Is the watermark present?")
	awgn_attackedImage, awgn_wpsnrWatermarkAttacked, awgn_decisionMade, awgn_std= bf.awgn_bf_best(name_originalImage, name_watermarkedImage, name_attackedImage)
	printer_of_best("awgn", "std", awgn_std, awgn_wpsnrWatermarkAttacked, awgn_attackedImage)
	print("BLUR: Is the watermark present?")
	blur_attackedImage, blur_wpsnrWatermarkAttacked, blur_decisionMade, blur_sigma = bf.blur_bf_best(name_originalImage, name_watermarkedImage, name_attackedImage)
	printer_of_best("blur", "sigma", blur_sigma, blur_wpsnrWatermarkAttacked, blur_attackedImage)
	print("JPEG: Is the watermark present?")
	jpeg_attackedImage, jpeg_wpsnrWatermarkAttacked, jpeg_decisionMade, jpeg_std = bf.jpeg_compression_bf_best(name_originalImage, name_watermarkedImage, name_attackedImage)
	printer_of_best("jpeg", "std", jpeg_std, jpeg_wpsnrWatermarkAttacked, jpeg_attackedImage)
	print("RESIZING: Is the watermark present?")
	resizing_attackedImage, resizing_wpsnrWatermarkAttacked, resizing_decisionMade, resizing_scale = bf.resizing_bf_best(name_originalImage, name_watermarkedImage, name_attackedImage)
	printer_of_best("resizing", "scale", resizing_scale, resizing_wpsnrWatermarkAttacked, resizing_attackedImage)
	print("MEDIAN: Is the watermark present?")
	median_attackedImage, median_wpsnrWatermarkAttacked, median_decisionMade, median_scale = bf.median_bf_best(name_originalImage, name_watermarkedImage, name_attackedImage)
	printer_of_best("median", "scale", median_scale, median_wpsnrWatermarkAttacked, median_attackedImage)
	print("SHARPENING basic bf ")
	sharpening_attackedImage, sharpening_wpsnrWatermarkAttacked, sharpening_decisionMade, sharpening_sigma, sharpening_alpha = bf.sharpening_pretty_basic_bf_best(name_originalImage, name_watermarkedImage, name_attackedImage)
	if hasattr(sharpening_attackedImage, "__len__"):
		print("BEST sharpening")
		print("sharpening_alpha=", sharpening_alpha)
		print("sharpening_sigma=", sharpening_sigma)
		print("sharpening_wpsnrWatermarkAttacked=", sharpening_wpsnrWatermarkAttacked)
	else:
		print("sharpening does not work")
	
	
	print("LOCALISED ATTACK", end = " ")
	attackedImage, wpsnrWatermarkAttacked, decisionMade=la.localised_attack(name_originalImage, name_watermarkedImage, name_attackedImage,
					sharpening_sigma=0.5, sharpening_alpha=0.4, awgn_std=20, median_kernel_size=3, flat_sensor=0)
	if decisionMade==0:
		print("with wpsnr=", wpsnrWatermarkAttacked)
	else:
		print("localised does not work")

	print("COMPOSITION ATTACK")
	attackedImage, wpsnr, decisionMade, std, sigma = bfa.awgn_blurring(name_originalImage, name_watermarkedImage, name_attackedImage)
	name_for_saving = name_attackedImage + "_awgn_blurring_std_" + str(std) + "_sigma_" + str(sigma) + "_wpsnr_" + str(wpsnr)+".bmp"
	attackedImage = name_attackedImage[:-4]+'_awgn_blurring.bmp'
	os.rename(attackedImage, name_for_saving)
	print("AWGN-Blurring-Attack with wpsnr = " + str(wpsnr))
	attackedImage, wpsnr, decisionMade, kernel, sigma = bfa.median_blurring(name_originalImage,name_watermarkedImage, name_attackedImage)
	name_for_saving = name_attackedImage + "_median_blurring_kernel_" + str(kernel) + "_sigma_" + str(sigma) + "_wpsnr_" + str(wpsnr)+".bmp"
	attackedImage = name_attackedImage[:-4] + '_median_blurring.bmp'
	os.rename(attackedImage, name_for_saving)
	print("Median-Blurring-Attack with wpsnr = " + str(wpsnr))
	attackedImage, wpsnr, decisionMade, scale, sigma = bfa.resizing_blurring(name_originalImage, name_watermarkedImage,name_attackedImage)
	name_for_saving = name_attackedImage + "_resizing_blurring_scale_" + str(scale) + "_sigma_" + str(sigma) + "_wpsnr_" + str(wpsnr)+".bmp"
	attackedImage = name_attackedImage[:-4]+'_resizing_blurring.bmp'
	os.rename(attackedImage, name_for_saving)
	print("Resizing-Blurring-Attack with wpsnr = " + str(wpsnr))
	attackedImage, wpsnr, decisionMade, std, kernel = bfa.awgn_median(name_originalImage,name_watermarkedImage,name_attackedImage)
	name_for_saving = name_attackedImage + "_awgn_median_std_" + str(std) + "_kernel_" + str(kernel) + "_wpsnr_" + str(wpsnr)+".bmp"
	attackedImage = name_attackedImage[:-4] + '_awgn_median.bmp'
	os.rename(attackedImage, name_for_saving)
	print("AWGN-Median-Attack with wpsnr = " + str(wpsnr))

#CODE TO BE DELETED FOR THE COMPETITION
#"""string_image = int(input("Enter name of the image without '.bmp': "))"""
#name_mark="./ef26420c.npy"
#"""name_originalImage = "../sample-images-roc/{num:04d}.bmp".format(num=string_image)"""
#string_image = input("Enter name of the image without '.bmp': ")
#
#name_originalImage = "./sample_images_roc/"+string_image+".bmp"
#
#"""
#Embedding
#watermarked = emb.embedding(name_originalImage, name_mark, alpha=5, name_output='watermarked.bmp', dim=8, step=20, max_splits=500,
#							min_splits=170, sub_size=6, Xi_exp = 0.2, Lambda_exp = 0.3, L_exp = 0.2, ceil = True)
#definitive_attack(name_originalImage, 'watermarked.bmp')
#"""
#name_watermarkedImage = "./embedded_images/"+string_image+".bmp"
#
#definitive_attack(name_originalImage, name_watermarkedImage, name_attackedImage)


#CODE TO BE MAINTAINTED FOR THE COMPETITION
name_images_competitors=["rollercoaster"]
# name_images_competitors=["buildings", "rollercoaster", "tree"]
# name_groups_competitors=["pixel"]
name_groups_competitors=["youshallnotmark"]
# name_groups_competitors=["ef26420c", "pixel", "youshallnotmark",
#						 "blitz", "omega", "howimetyourmark", "weusedlsb", "thebavarians",
#						 "theyarethesamepicture", "dinkleberg", "failedfouriertransform"]

start = time.time()
for name_image_to_be_attacked in name_images_competitors:
	for name_group_to_be_attacked in name_groups_competitors:
		print("I'm trying to attack " + name_group_to_be_attacked+ "_"+name_image_to_be_attacked+".bmp")

		name_watermarkedImage="./images_of_competition/"+ name_group_to_be_attacked+ "_"+name_image_to_be_attacked+".bmp"
		name_originalImage="./sample_images_roc/"+name_image_to_be_attacked+".bmp"
		name_attackedImage="./attacked/"+name_group_to_be_attacked+"/ef26420c_"+name_group_to_be_attacked+"_"+name_image_to_be_attacked+".bmp"
		definitive_attack(name_originalImage, name_watermarkedImage, name_attackedImage)
end = time.time()
print("the attacks take ", end-start, " seconds")




