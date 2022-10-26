import def_attack as da
import numpy as np
from scipy.fft import dct, idct
import hvs as hvs
import cv2
import matplotlib.pyplot as plt
import embedding_sub as emb
import psnr as psnr
import image_processing as ip
import detection_sub as det

def printer_of_best(nameOfAttack, nameOfvariable, valueOfvariable, wpsnr, attackedImage):
	if hasattr(attackedImage, "__len__"):
		print("BEST ", nameOfAttack)
		print(nameOfAttack, "_", nameOfvariable, "=", valueOfvariable)
		print(nameOfAttack, "_wpsnrWatermarkAttacked=", wpsnr)
	else:
		print(nameOfAttack, " does not work")
	

def definitive_attack(name_originalImage, name_watermarkedImage):
	#this function try first all the attacks alone, then try to combine them
	#it should be returned the best wpsnr
	print("AWGN: Is the watermark present?")
	awgn_attackedImage, awgn_wpsnrWatermarkAttacked, awgn_decisionMade, awgn_std= da.awgn_bf_best(name_originalImage, name_watermarkedImage)
	printer_of_best("awgn", "std", awgn_std, awgn_wpsnrWatermarkAttacked, awgn_attackedImage)

	print("BLUR: Is the watermark present?")
	blur_attackedImage, blur_wpsnrWatermarkAttacked, blur_decisionMade, blur_sigma = da.blur_bf_best(name_originalImage, name_watermarkedImage)
	printer_of_best("blur", "sigma", blur_sigma, blur_wpsnrWatermarkAttacked, blur_attackedImage)

	print("JPEG: Is the watermark present?")
	jpeg_attackedImage, jpeg_wpsnrWatermarkAttacked, jpeg_decisionMade, jpeg_std = da.jpeg_compression_bf_best(name_originalImage, name_watermarkedImage)
	printer_of_best("jpeg", "std", jpeg_std, jpeg_wpsnrWatermarkAttacked, jpeg_attackedImage)

	print("RESIZING: Is the watermark present?")
	resizing_attackedImage, resizing_wpsnrWatermarkAttacked, resizing_decisionMade, resizing_scale = da.resizing_bf_best(name_originalImage, name_watermarkedImage)
	printer_of_best("resizing", "scale", resizing_scale, resizing_wpsnrWatermarkAttacked, resizing_attackedImage)

	print("MEDIAN: Is the watermark present?")
	median_attackedImage, median_wpsnrWatermarkAttacked, median_decisionMade, median_scale = da.median_bf_best(name_originalImage, name_watermarkedImage)
	printer_of_best("median", "scale", median_scale, median_wpsnrWatermarkAttacked, median_attackedImage)

#	print("SHARPENING")
#	sharpening_attackedImage, sharpening_wpsnrWatermarkAttacked, sharpening_decisionMade, sharpening_sigma, sharpening_alpha = da.sharpening_bf_best(name_originalImage, name_watermarkedImage)
#	if hasattr(sharpening_attackedImage, "__len__"):
#		print("BEST sharpening")
#		print("sharpening_alpha=", sharpening_alpha)
#		print("sharpening_sigma=", sharpening_sigma)
#		print("sharpening_wpsnrWatermarkAttacked=", sharpening_wpsnrWatermarkAttacked)
#	else:
#		print("sharpening does not work")

string_image = input("Enter name of the image without '.bmp': ")
name_originalImage = str(string_image)+'.bmp'
MARK = np.load('ef26420c.npy')
mark = np.array([(-1) ** m for m in MARK])
#Embedding
watermarked = emb.embedding(name_originalImage, mark, alpha = 10, name_output = 'watermarked.bmp')
originalImage = cv2.imread(name_originalImage, 0)
watermarkedImage = cv2.imread('watermarked.bmp', 0)


definitive_attack(name_originalImage, 'watermarked.bmp')






