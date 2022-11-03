#After embedded the image, I want to test how robusts the watermark is against the attacks
#So I am interested only in that value of std that break the watermark
#Remember: we want that the watermark will be not present
import image_processing as ip
import detection_ef26420c as det
import numpy as np
import cv2


#AWGN
def awgn_bf(name_originalImage, name_watermarkedImage, std_min, std_max, std_step, seed, direction):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	name_attackedImage='./attacked/attacked_awgn.bmp'
	if direction==1:
	#go this way-> 
		for std in np.arange(std_min, std_max+std_step, std_step):
			attackedImage=ip.awgn(watermarkedImage, std, seed)
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			#I want to destroy the watermark
			if decisionMade==0:
#				print("No: std=", std,"-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, std
			if wpsnrWatermarkAttacked<35:
				print("wpsnrWatermarkAttacked is less than 35...")
				break
#			print("Yes: std=", std, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	elif direction==-1:
	#go the other way <-
		for std in np.flip(np.arange(std_min-std_step,std_max+std_step, std_step)):
			attackedImage=ip.awgn(watermarkedImage, std, seed)
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			#I want to find the watermark
			if decisionMade==1:
#				print("Yes: std=", std, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, std
#			print("No: std=", std, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	return 0, "awgn does not work", "awgn does not work", "awgn does not work"

def awgn_bf_best(name_originalImage, name_watermarkedImage):
	#it seems that the seed does not influence much the attack
	attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(name_originalImage, name_watermarkedImage, 0, 80 , 1   , 123, 1)
	if hasattr(attackedImage, "__len__"):
		#I'm sure that with std-1 the watermark will be present as I want
		attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(name_originalImage, name_watermarkedImage, std-1, std	, 0.1 , 123, -1)
		if hasattr(attackedImage, "__len__"):
			#I'm sure that with std+0.1 the watermark will be NOT present as I want
			attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(name_originalImage, name_watermarkedImage, std  , std+0.1, 0.01, 123, 1)
			if hasattr(attackedImage, "__len__"):
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, std
	return 0, "awgn does not work", "awgn does not work", "awgn does not work"



#BLUR
def blur_bf(name_originalImage, name_watermarkedImage, sigma_min, sigma_max, sigma_step, direction):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	name_attackedImage='./attacked/attacked_blur.bmp'
	if direction==1:
		#go this way-> 
		for sigma in np.arange(sigma_min, sigma_max+sigma_step, sigma_step):
			attackedImage=ip.blur(watermarkedImage, sigma)
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			#I want to destroy the watermark
			if decisionMade==0:
#				print("No: sigma=", sigma, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)				  
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma
#			print("Yes: sigma=", sigma,"-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	elif direction==-1:
		#go the other way <-
		for sigma in np.flip(np.arange(sigma_min-sigma_step,sigma_max, sigma_step)):
			attackedImage=ip.blur(watermarkedImage, sigma)
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			#I want to find the watermark
			if decisionMade==1:
#				print("Yes: sigma=", sigma, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked) 
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma
#			print("No: sigma=", sigma, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	return 0, "blur does not work", "blur does not work", "blur does not work"
  
def blur_bf_best(name_originalImage, name_watermarkedImage):
	#it seems that the seed does not influence much the attack
	attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma = blur_bf(name_originalImage, name_watermarkedImage, 0	, 50	 , 1   , 1)
	if hasattr(attackedImage, "__len__"):
		#I'm sure that with std-1 the watermark will be present as I want
		attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma = blur_bf(name_originalImage, name_watermarkedImage, sigma-1, sigma	, 0.1 , -1)
		if hasattr(attackedImage, "__len__"):
			#I'm sure that with std+0.1 the watermark will be NOT present as I want
			attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma = blur_bf(name_originalImage, name_watermarkedImage, sigma  , sigma+0.1, 0.01, 1)
			if hasattr(attackedImage, "__len__"):
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma
	return 0, "blur does not work", "blur does not work", "blur does not work"



#JPEG_COMPRESSION
def jpeg_compression_bf(name_originalImage, name_watermarkedImage, qf_min, qf_max, qf_step, direction):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	name_attackedImage='./attacked/attacked_jpeg.bmp'
	if direction==-1:
		#go the other way <-
		for qf in np.flip(np.arange(qf_min-qf_step,qf_max, qf_step)):
			attackedImage=ip.jpeg_compression(watermarkedImage, qf)
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			#I want to find the watermark
			if decisionMade==0:
#				print("No: qf=", qf, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked) 
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf
#			print("Yes: qf=", qf, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	if direction==1:
		#go on this way ->
		for qf in np.arange(qf_min-qf_step,qf_max, qf_step):
			attackedImage=ip.jpeg_compression(watermarkedImage, qf)
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			#I want to find the watermark
			if decisionMade==1:
#				print("Yes: qf=", qf, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked) 
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf
#			print("No: qf=", qf, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	return 0, "jpeg_compression does not work", "jpeg_compression does not work", "jpeg_compression does not work"
  
def jpeg_compression_bf_best(name_originalImage, name_watermarkedImage):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	#it seems that the seed does not influence much the attack
	attackedImage, wpsnrWatermarkAttacked, decisionMade, qf = jpeg_compression_bf(name_originalImage, name_watermarkedImage, 1   , 70	, 5   , -1)
	if hasattr(attackedImage, "__len__"):
		attackedImage, wpsnrWatermarkAttacked, decisionMade, qf = jpeg_compression_bf(name_originalImage, name_watermarkedImage, qf   , qf+5	, 1   , 1)
		if hasattr(attackedImage, "__len__"):
			return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf
	return 0, "jpeg_compression does not work", "jpeg_compression does not work", "jpeg_compression does not work"



#RESIZING
def resizing_bf(name_originalImage, name_watermarkedImage, scale_min, scale_max, scale_step, direction):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	name_attackedImage='./attacked/attacked_resizing.bmp'
	if direction==1:
		#go this way-> 
		for scale in np.arange(scale_min, scale_max+scale_step, scale_step):
			attackedImage=ip.resizing(watermarkedImage, scale)
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			#I want to destroy the watermark
			if decisionMade==1:
#				print("Yes: scale=", scale, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)				  
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale
#			print("No: scale=", scale,"-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	elif direction==-1:
		#go the other way <-
		for scale in np.flip(np.arange(scale_min-scale_step,scale_max, scale_step)):
			if scale==0:
				return attackedImage, wpsnrWatermarkAttacked, 0, 0
			attackedImage=ip.resizing(watermarkedImage, scale)
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			#I want to find the watermark
			if decisionMade==0:
#				print("No: scale=", scale, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked) 
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale
#			print("Yes: scale=", scale, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	return 0, "resizing does not work", "resizing does not work", "resizing does not work"


def resizing_bf_best(name_originalImage, name_watermarkedImage):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(name_originalImage, name_watermarkedImage, 0, 4	 , 2   , -1)
	if hasattr(attackedImage, "__len__"):
		#I'm sure that with scale-1 the watermark will not be present as I want
		if scale==0:
			attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(name_originalImage, name_watermarkedImage, scale+0.1, scale+1	, 0.1 , 1)
		else:
			attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(name_originalImage, name_watermarkedImage, scale, scale+1	, 0.1 , 1)
		if hasattr(attackedImage, "__len__"):
			#in sttackedImage there is the watermark
			attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(name_originalImage, name_watermarkedImage, scale-0.1, scale, 0.01 , -1)
			if hasattr(attackedImage, "__len__"):
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale
	return 0, "resizing does not work", "resizing does not work", "resizing does not work"


#MEDIAN
def median_bf(name_originalImage, name_watermarkedImage, kernel_size_min, kernel_size_max, kernel_size_step): #Remark: kernel_size must be odd integer
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	name_attackedImage='./attacked/attacked_median.bmp'
	for kernel_size in np.arange(kernel_size_min, kernel_size_max+kernel_size_step, kernel_size_step):
		attackedImage = ip.median(watermarkedImage, kernel_size)
		cv2.imwrite(name_attackedImage, attackedImage)
		decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
		#I want to destroy the watermark
		if decisionMade==0 and wpsnrWatermarkAttacked<1000:
#			print("No: kernel_size=", kernel_size, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
			return attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size
	return 0, "median does not work", "median does not work", "median does not work"

def median_bf_best(name_originalImage, name_watermarkedImage):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	#it seems that the seed does not influence much the attack
	attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size = median_bf(name_originalImage, name_watermarkedImage, 3   ,  101   , 2)
	if hasattr(attackedImage, "__len__"):
		return attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size
	return 0, "median does not work", "median does not work", "median does not work"



#SHARPENING
def sharpening_bf(name_originalImage, name_watermarkedImage, sigma_min, sigma_max, sigma_step, alpha_min, alpha_max, alpha_step):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	name_attackedImage='./attacked/attacked_sharpening.bmp'
	listwpsnrwatermark=[]
	wpsnrValue=[]
	#Evaluation of attacks
	for sigma in np.arange(sigma_min, sigma_max+sigma_step, sigma_step):# """We have to analyse which are the best values"""
		for alpha in np.arange(alpha_min, alpha_max+alpha_step, alpha_step):
			attackedImage=ip.sharpening(watermarkedImage, sigma, alpha) #this it the image attacked
			cv2.imwrite(name_attackedImage, attackedImage)
			decisionMade, wpsnrWatermarkAttacked = det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
			if decisionMade==0:
				listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha])
				break
			if wpsnrWatermarkAttacked < 30:
#				print("sigma=", sigma, "alpha=", alpha, "-> wpsnrWatermarkAttacked less then 30...")
				break
#			print("Yes: sigma=", sigma, "alpha=", alpha, "-> wpsnrWatermarkAttacked=", wpsnrWatermarkAttacked)
	#Unfortunately it takes to much time to search wrt the three best values.
	#So, we will do wrt only to the best
	# if time==1 or time==2:
	#   toBeReturned=[]
	#   for i in range(3):
	#	 sublista=[sublist[1] for sublist in listwpsnrwatermark]
	#	 indice=sublista.index(max(sublista)) #return the index of max wpsnr
	#	 toBeReturned.append(listwpsnrwatermark[indice])
	#	 listwpsnrwatermark.pop(indice)
	#   return toBeReturned
	if len(listwpsnrwatermark)==0:
		return 0, "list==0", "list==0", "list==0", "list==0"
	else:
		sublista=[sublist[1] for sublist in listwpsnrwatermark]
		indice=sublista.index(max(sublista))
		attackedImage=listwpsnrwatermark[indice][0]
		wpsnrWatermarkAttacked=listwpsnrwatermark[indice][1]
		decisionMade=listwpsnrwatermark[indice][2]
		sigma=listwpsnrwatermark[indice][3]
		alpha=listwpsnrwatermark[indice][4]
		return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha
	return 0, "Something wrong", "Something wrong", "Something wrong", "Something wrong"

def sharpening_bf_best(name_originalImage, name_watermarkedImage):
	originalImage = cv2.imread(name_originalImage, 0)
	watermarkedImage = cv2.imread(name_watermarkedImage, 0)
	#sharpening search
	attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha = sharpening_bf(name_originalImage, name_watermarkedImage,0.4, 1.0, 0.1,0.4, 1.0, 0.1)
	if hasattr(attackedImage, "__len__"):
		#sharpening search attackedImage, sigma, alpha
		attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha = sharpening_bf(name_originalImage, name_watermarkedImage, sigma-0.1, sigma, 0.01, alpha-0.1, alpha, 0.01)
		if hasattr(attackedImage, "__len__"):
			#sharpening search
			attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha = sharpening_bf(name_originalImage, name_watermarkedImage, sigma-0.01, sigma, 0.001, alpha-0.01, alpha, 0.001)
			if hasattr(attackedImage, "__len__"):
				#Definitve decision
				return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha
	return 0, "Something wrong in general", "Something wrong in general", "Something wrong in general"





