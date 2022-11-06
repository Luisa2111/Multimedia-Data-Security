import hvs_lambda as hvs
import cv2
import matplotlib.pyplot as plt
import os

import image_processing as ip
import psnr as psnr
import bf_attack as bf
import numpy as np
import detection_caller as det_c

#REMARK: since jpeg_compression and resizing change the dimensions of the squares, we can't combine them in a localised way
def localised_attack(name_originalImage, name_watermarkedImage, name_attackedImage, awgn_std=21, blur_sigma=1.3, median_kernel_size=5, sharpening_sigma=0.5, sharpening_alpha=0.5, flat_sensor=0):
	print("with flat sensibility:", flat_sensor)
	print("awgn_std", awgn_std)
	print("blur_sigma", blur_sigma)
	print("median_kernel_size", median_kernel_size)
	print("sharpening_sigma", sharpening_sigma)
	print("sharpening_alpha", sharpening_alpha)
	
	
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	blocks = hvs.hvs_blocks(originalImage)
	
	attackedImage=watermarkedImage.copy()
	
	
	for row in range(16): #8 is the number of rows
		for column in range(16):
			if blocks[row, column]<=flat_sensor: #flat squares 
				attackedImage[32*row:32*(row+1),32*column: 32*(column+1)]=ip.awgn(attackedImage[32*row:32*(row+1),32*column:
																												32*(column+1)], 
																					awgn_std, 123)
				
			else: #textured squares
				attackedImage[32*row:32*(row+1),32*column: 32*(column+1)]=ip.sharpening(attackedImage[32*row:32*(row+1), 
																											32*column:32*(column+1)],
																						 sharpening_sigma, sharpening_alpha)

	
#	attackedImage=ip.sharpening(attackedImage, sharpening_sigma, sharpening_alpha)
#	attackedImage=ip.blur(attackedImage, blur_sigma)
	
	name_attackedImage=name_attackedImage[:-4]+'LA.bmp'
	cv2.imwrite(name_attackedImage, attackedImage)
	
	decisionMade, wpsnrWatermarkAttacked = det_c.detection_caller(name_originalImage, name_watermarkedImage, name_attackedImage)
	
	os.rename(name_attackedImage, name_attackedImage[:-4]+str(wpsnrWatermarkAttacked)[:5]+'.bmp')
	
#	print(blocks)
#	plt.figure(figsize=(15, 6))
#	plt.subplot(111)
#	plt.title('Block')
#	plt.imshow(blocks)
#	plt.show()
	
	return attackedImage, wpsnrWatermarkAttacked, decisionMade
	
	

