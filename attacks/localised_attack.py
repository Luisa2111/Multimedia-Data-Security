import hvs_lambda as hvs
import cv2
import matplotlib.pyplot as plt
import image_processing as ip
import psnr as psnr
import bf_attack as bf
import numpy as np
from scipy.fft import dct, idct
import embedding_ef26420c as emb
import detection_ef26420c as det

def localised_attack(name_originalImage, name_watermarkedImage, awgn_std=21, blur_sigma=1.3, jpeg_compression_qf=5, resizing_scale=0.32, median_kernel_size=5, sharpening_sigma=0.5, sharpening_alpha=0.5):
	print("awgn_std", awgn_std)
	print("blur_sigma", blur_sigma)
	print("jpeg_compression_qf", jpeg_compression_qf)
	print("resizing_scale", resizing_scale)
	print("kernel_size_median", median_kernel_size)
	print("sharpening_sigma", sharpening_sigma)
	print("sharpening_alpha", sharpening_alpha)
	
	
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	blocks = hvs.hvs_blocks(originalImage)
	
	attackedImage=watermarkedImage.copy()
	
	for row in range(16): #8 is the number of rows
		for column in range(16):
			if blocks[row, column]==0: #flat squares 
				attackedImage[64*row:64*(row+1),64*column: 64*(column+1)]=ip.awgn(attackedImage[64*row:64*(row+1),64*column:
																												64*(column+1)], 
																					awgn_std, 123)
			else: #textured squares
				attackedImage[64*row:64*(row+1),64*column: 64*(column+1)]=ip.sharpening(attackedImage[64*row:64*(row+1), 
																											64*column:64*(column+1)],
																						 sharpening_sigma, sharpening_alpha)
	
	cv2.imwrite("./attacked/hvsBlock.bmp", )
#	attackedImage=ip.sharpening(attackedImage, sharpening_sigma, sharpening_alpha)
#	attackedImage=ip.blur(attackedImage, blur_sigma)
	
	name_attackedImage='./attacked/attackedImageBlocks.bmp'
	cv2.imwrite(name_attackedImage, attackedImage)
	
	decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
	
	return attackedImage, wpsnrWatermarkAttacked, decisionMade
	
	

