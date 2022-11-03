import numpy as np
from scipy.fft import dct, idct
import hvs_lambda as hvs
import cv2
import matplotlib.pyplot as plt
import embedding_ef26420c as emb

import psnr as psnr
import bf_attack as bf
import localised_attack as la

def printer_of_best(nameOfAttack, nameOfvariable, valueOfvariable, wpsnr, attackedImage):
	if hasattr(attackedImage, "__len__"):
		print("BEST ", nameOfAttack)
		print(nameOfAttack, "_", nameOfvariable, "=", valueOfvariable)
		print(nameOfAttack, "_wpsnrWatermarkAttacked=", wpsnr)
	else:
		print(nameOfAttack, " does not work")
	

def definitive_attack(name_originalImage, name_watermarkedImage):
	#this function try first all the attacks alone
	#then try to localise tham
	#then try to combine them
	
	#it should be returned the best wpsnr
#	print("AWGN: Is the watermark present?")
#	awgn_attackedImage, awgn_wpsnrWatermarkAttacked, awgn_decisionMade, awgn_std= bf.awgn_bf_best(name_originalImage, name_watermarkedImage)
#	printer_of_best("awgn", "std", awgn_std, awgn_wpsnrWatermarkAttacked, awgn_attackedImage)
#	print("BLUR: Is the watermark present?")
#	blur_attackedImage, blur_wpsnrWatermarkAttacked, blur_decisionMade, blur_sigma = bf.blur_bf_best(name_originalImage, name_watermarkedImage)
#	printer_of_best("blur", "sigma", blur_sigma, blur_wpsnrWatermarkAttacked, blur_attackedImage)
#	print("JPEG: Is the watermark present?")
#	jpeg_attackedImage, jpeg_wpsnrWatermarkAttacked, jpeg_decisionMade, jpeg_std = bf.jpeg_compression_bf_best(name_originalImage, name_watermarkedImage)
#	printer_of_best("jpeg", "std", jpeg_std, jpeg_wpsnrWatermarkAttacked, jpeg_attackedImage)
#	print("RESIZING: Is the watermark present?")
#	resizing_attackedImage, resizing_wpsnrWatermarkAttacked, resizing_decisionMade, resizing_scale = bf.resizing_bf_best(name_originalImage, name_watermarkedImage)
#	printer_of_best("resizing", "scale", resizing_scale, resizing_wpsnrWatermarkAttacked, resizing_attackedImage)
#	print("MEDIAN: Is the watermark present?")
#	median_attackedImage, median_wpsnrWatermarkAttacked, median_decisionMade, median_scale = bf.median_bf_best(name_originalImage, name_watermarkedImage)
#	printer_of_best("median", "scale", median_scale, median_wpsnrWatermarkAttacked, median_attackedImage)


#	print("SHARPENING")
#	sharpening_attackedImage, sharpening_wpsnrWatermarkAttacked, sharpening_decisionMade, sharpening_sigma, sharpening_alpha = bf.sharpening_bf_best(name_originalImage, name_watermarkedImage)
#	if hasattr(sharpening_attackedImage, "__len__"):
#		print("BEST sharpening")
#		print("sharpening_alpha=", sharpening_alpha)
#		print("sharpening_sigma=", sharpening_sigma)
#		print("sharpening_wpsnrWatermarkAttacked=", sharpening_wpsnrWatermarkAttacked)
#	else:
#		print("sharpening does not work")
	
	print("LOCALISED ATTACK")
	attackedImage, wpsnrWatermarkAttacked, decisionMade=la.localised_attack(name_originalImage, name_watermarkedImage,
					sharpening_sigma=0.47, sharpening_alpha=0.25, awgn_std=10)
	if decisionMade==0:
		print("with wpsnr=", wpsnrWatermarkAttacked)
	else:
		print("localised does not work")	
	

#string_image = int(input("Enter name of the image without '.bmp': "))
name_mark="./ef26420c.npy"
#name_originalImage = "../sample-images-roc/{num:04d}.bmp".format(num=string_image)
string_image = input("Enter name of the image without '.bmp': ")

name_originalImage = "./sample_images_roc/"+string_image+".bmp"

#Embedding
#watermarked = emb.embedding(name_originalImage, name_mark, alpha=5, name_output='watermarked.bmp', dim=8, step=20, max_splits=500,
#							min_splits=170, sub_size=6, Xi_exp = 0.2, Lambda_exp = 0.3, L_exp = 0.2, ceil = True)
#definitive_attack(name_originalImage, 'watermarked.bmp')

name_watermarkedImage = "./embedded_images/"+string_image+".bmp"

definitive_attack(name_originalImage, name_watermarkedImage)


