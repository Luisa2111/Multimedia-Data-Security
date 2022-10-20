#After embedded the image, I want to test how robusts the watermark is against the attacks
#So I am interested only in that value of std that break the watermark
#Remember: we want that the watermark will be not present

import numpy as np

def awgn_bf(originalImage, watermarkedImage, std_min, std_max, std_step, seed):
  listwpsnrwatermark=[]

  for std in np.arange(std_min, std_max, std_step):# """We have to analyse which are the best values"""
    attackedImage=awgn(watermarkedImage, std, seed) #this it the image attacked
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, std])

  return listwpsnrwatermark

  
def blur_bf(originalImage, watermarkedImage, sigma_min, sigma_max, sigma_step):
  listwpsnrwatermark=[]

  for sigma in np.arange(sigma_min, sigma_max, sigma_step):# """We have to analyse which are the best values"""
    attackedImage=blur(watermarkedImage, sigma) #this it the image attacked
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma])

  return listwpsnrwatermark
  

  
def sharpening_bf(originalImage, watermarkedImage, sigma_min, sigma_max, sigma_step, alpha_min, alpha_max, alpha_step):
  listwpsnrwatermark=[]

  for sigma in np.arange(sigma_min, sigma_max, sigma_step):# """We have to analyse which are the best values"""
    for alpha in np.arange(alpha_min, alpha_max, alpha_step):
      attackedImage=sharpening(watermarkedImage, sigma, alpha) #this it the image attacked
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
      listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma, alpha])

  return listwpsnrwatermark
  

def jpeg_compression_bf(originalImage, watermarkedImage, qf_max):
  listwpsnrwatermark=[]

  for qf in range(qf_max):# """We have to analyse which are the best values"""
    attackedImage=jpeg_compression(watermarkedImage, qf) #this it the image attacked
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, qf])

  return listwpsnrwatermark


def resizing_bf(originalImage, watermarkedImage, scale_min, scale_max, scale_step):
  listwpsnrwatermark=[]

  for scale in np.arange(scale_min, scale_max, scale_step):# """We have to analyse which are the best values"""
    if scale==0:
      listwpsnrwatermark.append([originalImage, 9999999, 1, 0])
    else:
      attackedImage=resizing(watermarkedImage, scale) #this it the image attacked
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
      listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, scale])
    
  return listwpsnrwatermark


def median_bf(originalImage, watermarkedImage, scale_min, scale_max, scale_step):
  listwpsnrwatermark=[]

  for kernel_size in np.arange(scale_min, scale_max, scale_step):# """We have to analyse which are the best values"""
    attackedImage=median_attack(watermarkedImage, kernel_size) #this it the image attacked
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size])
    
  return listwpsnrwatermark
