#After embedded the image, I want to test how robusts the watermark is against the attacks
#So I am interested only in that value of std that break the watermark
#Remember: we want that the watermark will be not present
import image_processing as ip
import detection as det
import numpy as np


#AWGN
def awgn_bf(originalImage, watermarkedImage, std_min, std_max, std_step, seed, direction):
  if direction==1:
    #go this way ->
    for std in np.arange(std_min, std_max+std_step, std_step):
      attackedImage=ip.awgn(watermarkedImage, std, seed)
      decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
      #I want to destroy the watermark
      if decisionMade==0:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, std
  elif direction==-1:
    #go the other way <-
    for std in np.flip(np.arange(std_min-std_step,std_max+std_step, std_step)):
      attackedImage=ip.awgn(watermarkedImage, std, seed)
      decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
      #I want to find the watermark
      if decisionMade==1:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, std
  return "awgn does not work"

def awgn_bf_best(originalImage, watermarkedImage):
  #it seems that the seed does not influence much the attack
  attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(originalImage, watermarkedImage, 0    , 50     , 1   , 123, 1)
  #I'm sure that with std-1 the watermark will be present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(originalImage, watermarkedImage, std-1, std    , 0.1 , 123, -1)
  #I'm sure that with std+0.1 the watermark will be NOT present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(originalImage, watermarkedImage, std  , std+0.1, 0.01, 123, 1)
  return attackedImage, wpsnrWatermarkAttacked, decisionMade, std



#BLUR
def blur_bf(originalImage, watermarkedImage, sigma_min, sigma_max, sigma_step, direction):
  if direction==1:
    #go this way ->
    for sigma in np.arange(sigma_min, sigma_max+sigma_step, sigma_step):
      attackedImage=ip.blur(watermarkedImage, sigma)
      decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
      #I want to destroy the watermark
      if decisionMade==0:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma
  elif direction==-1:
    #go the other way <-
    for sigma in np.flip(np.arange(sigma_min-sigma_step,sigma_max+sigma_step, sigma_step)):
      attackedImage=ip.blur(watermarkedImage, sigma)
      decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
      #I want to find the watermark
      if decisionMade==1:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma
  return "blur does not work"
  
def blur_bf_best(originalImage, watermarkedImage):
  #it seems that the seed does not influence much the attack
  attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma = blur_bf(originalImage, watermarkedImage, 0    , 50     , 1   , 1)
  #I'm sure that with std-1 the watermark will be present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma = blur_bf(originalImage, watermarkedImage, sigma-1, sigma    , 0.1 , -1)
  #I'm sure that with std+0.1 the watermark will be NOT present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma = blur_bf(originalImage, watermarkedImage, sigma  , sigma+0.1, 0.01, 1)
  return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma
  


#SHARPENING
def sharpening_bf(originalImage, watermarkedImage, sigma_min, sigma_max, sigma_step, alpha_min, alpha_max, alpha_step):
  listwpsnrwatermark=[]
  wpsnrValue=[]
  #Evaluation of attacks
  for sigma in np.arange(sigma_min, sigma_max+sigma_step, sigma_step):# """We have to analyse which are the best values"""
    for alpha in np.arange(alpha_min, alpha_max+alpha_step, alpha_step):
      attackedImage=ip.sharpening(watermarkedImage, sigma, alpha) #this it the image attacked
      decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
      if decisionMade==0:
        listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha])
        break
  #Unfortunately it takes to much time to search wrt the three best values.
  #So, we will do wrt only to the best
  # if time==1 or time==2:
  #   toBeReturned=[]
  #   for i in range(3):
  #     sublista=[sublist[1] for sublist in listwpsnrwatermark]
  #     indice=sublista.index(max(sublista)) #return the index of max wpsnr
  #     toBeReturned.append(listwpsnrwatermark[indice])
  #     listwpsnrwatermark.pop(indice)
  #   return toBeReturned
  if len(listwpsnrwatermark)==0:
    return "SOMETHING WENT WRONG"
  else:
    sublista=[sublist[1] for sublist in listwpsnrwatermark]
    indice=sublista.index(max(sublista))
    attackedImage=listwpsnrwatermark[indice][0]
    wpsnrWatermarkAttacked=listwpsnrwatermark[indice][1]
    decisionMade=listwpsnrwatermark[indice][2]
    sigma=listwpsnrwatermark[indice][3]
    alpha=listwpsnrwatermark[indice][4]
    return attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha
  return "SOMETHING WENT WRONG"

def sharpening_bf_best(originalImage, watermarkedImage):
  #First search
  first_attackedImage, first_wpsnrWatermarkAttacked, first_decisionMade, first_sigma, first_alpha=sharpening_bf(originalImage, watermarkedImage,    0.2, 1.0, 0.1,    0.2, 1.0, 0.1)
  #Second search attackedImage, sigma, alpha
  second_attackedImage, second_wpsnrWatermarkAttacked, second_decisionMade, second_sigma, second_alpha=sharpening_bf(originalImage, watermarkedImage, first_sigma-0.1, first_sigma, 0.01, first_alpha-0.1, first_alpha, 0.01)
  #Third search
  third_attackedImage, third_wpsnrWatermarkAttacked, third_decisionMade, third_sigma, third_alpha=sharpening_bf(originalImage, watermarkedImage, second_sigma-0.01, second_sigma, 0.001, second_alpha-0.01, second_alpha, 0.001)
  # #Definitve decision
  return third_attackedImage, third_wpsnrWatermarkAttacked, third_decisionMade, third_sigma, third_alpha



#JPEG_COMPRESSION
def jpeg_compression_bf(originalImage, watermarkedImage, qf_min, qf_max, qf_step, direction):
  if direction==1:
    #go this way ->
    for qf in np.arange(qf_min, qf_max+qf_step, qf_step):
      attackedImage=ip.jpeg_compression(watermarkedImage, qf)
      decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
      #I want to destroy the watermark
      if decisionMade==0:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf
  elif direction==-1:
    #go the other way <-
    for qf in np.flip(np.arange(qf_min-qf_step,qf_max+qf_step, qf_step)):
      attackedImage=jpeg_compression(watermarkedImage, qf)
      decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
      #I want to find the watermark
      if decisionMade==1:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf
  return "blur does not work"
  
def jpeg_compression_bf_best(originalImage, watermarkedImage):
  #it seems that the seed does not influence much the attack
  attackedImage, wpsnrWatermarkAttacked, decisionMade, qf = jpeg_compression_bf(originalImage, watermarkedImage, 0   , 50    , 1   , 1)
  #I'm sure that with std-1 the watermark will be present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, qf = jpeg_compression_bf(originalImage, watermarkedImage, qf-1, qf    , 0.1 , -1)
  #I'm sure that with std+0.1 the watermark will be NOT present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, qf = jpeg_compression_bf(originalImage, watermarkedImage, qf  , qf+0.1, 0.01, 1)
  return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf



#RESIZING
def resizing_bf(originalImage, watermarkedImage, scale_min, scale_max, scale_step, direction):
  if direction==1:
    #go this way ->
    for scale in np.arange(scale_min, scale_max+scale_step, scale_step):
      if scale!=0:
        attackedImage=ip.resizing(watermarkedImage, scale)
        decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
        #I want to destroy the watermark
        if decisionMade==0:
          return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale
  elif direction==-1:
    #go the other way <-
    for scale in np.flip(np.arange(scale_min-scale_step, scale_max+scale_step, scale_step)):
      if scale!=0:
        attackedImage=ip.resizing(watermarkedImage, scale)
        decisionMade, wpsnrWatermarkAttacked = det.detection(originalImage, watermarkedImage, attackedImage, MARK)
        #I want to find the watermark
        if decisionMade==1:
          return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale
  print("resizing does not work")
  return 0

def resizing_bf_best(originalImage, watermarkedImage):
  attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(originalImage, watermarkedImage, 1    , 50     , 1   , 1)
  #I'm sure that with scale-1 the watermark will be present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(originalImage, watermarkedImage, scale-1, scale    , 0.1 , -1)
  #I'm sure that with scale+0.1 the watermark will be NOT present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(originalImage, watermarkedImage, scale  , scale+0.1, 0.01, 1)
  #I'm sure that with scale-1 the watermark will be present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(originalImage, watermarkedImage, scale-0.01, scale    , 0.001 , -1)
  #I'm sure that with scale+0.1 the watermark will be NOT present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = resizing_bf(originalImage, watermarkedImage, scale  , scale+0.001, 0.0001, 1)
  return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale



#MEDIAN
def median_bf(originalImage, watermarkedImage, scale_min, scale_max, scale_step): #Remark: scale must be odd integer
  for scale in np.arange(scale_min, scale_max+scale_step, scale_step):
    attackedImage=ip.median(watermarkedImage, scale)
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage, MARK)
    #I want to destroy the watermark
    if decisionMade==0 and wpsnrWatermarkAttacked<1000:
      return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale
  return "blur does not work"
  
def median_bf_best(originalImage, watermarkedImage):
  #it seems that the seed does not influence much the attack
  attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = median_bf(originalImage, watermarkedImage, 3   ,  101   , 2)
  return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale


