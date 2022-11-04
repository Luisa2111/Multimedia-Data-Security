import cv2
import image_processing as ip
import psnr as psnr
import numpy as np
import embedding_ef26420c as emb
import detection_ef26420c as det
import os

def copositeAttack(name_originalImage, name_watermarkedImage, name_attackedImage):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	"""
	decide:
		awgn
		blur
		sharpening
		median
		resizing
		jpeg_compression
	"""
	
	flag=True
	while flag:
		whichAttack=int(input("which attack do you want to apply? \n 1-> awgn \n 2-> blur \n 3-> median \n 4-> sharpening \n 5-> resizing \n 6-> jpeg_compression \n 7-> exit composite attack \n"))
		if whichAttack==1:
			awgn_std=int(input("how big is std for awgn? (hint:[0, 50]) "))
			watermarkedImage=ip.awgn(watermarkedImage, awgn_std, 123)
		elif whichAttack==2:
			blur_sigma=float(input("how big is sigma for blur? (hint: [0.0, 1.0]) "))
			watermarkedImage=ip.blur(watermarkedImage, blur_sigma)
		elif whichAttack==3:
			median_kernel_size=int(input("how big is kernel_size for median? odd number "))
			watermarkedImage=ip.median(watermarkedImage, median_kernel_size)
		elif whichAttack==4:
			sharpening_sigma=float(input("how big is sigma for sharpening? (hint: [0.2, 2.0]) "))
			sharpening_alpha=float(input("how big is alpha for sharpening? (hint: [0.2, 2.0]) "))
			watermarkedImage=ip.sharpening(watermarkedImage, sharpening_sigma, sharpening_alpha)
		elif whichAttack==5:
			resizing_scale=float(input("how big is scale for resizing? (hint: [0.1, 2.0]) "))
			watermarkedImage=ip.resizing(watermarkedImage, resizing_scale)
		elif whichAttack==6:
			jpeg_compression_qf=int(input("how big is qf for jpeg_compression? (hint: [1, 10]) "))
			watermarkedImage=ip.jpeg_compression(watermarkedImage, jpeg_compression_qf)
		elif whichAttack==7:
			break
		
		name_attackedImage=name_attackedImage[:-4]+'CA.bmp'
		cv2.imwrite(name_attackedImage, watermarkedImage)
		
		decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
		if decisionMade:
			print("the watermark is still present")
			print("and the wpsnr is: ", wpsnrWatermarkAttacked)
		else:
			print("the watermark is no more present")
			print("and the wpsnr is: ", wpsnrWatermarkAttacked)
			os.rename(name_attackedImage, name_attackedImage[:-4]+str(wpsnrWatermarkAttacked)[:5]+'.bmp')
	
	
	return watermarkedImage, wpsnrWatermarkAttacked, decisionMade
