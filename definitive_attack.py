import def_attack
import numpy as np
from scipy.fft import dct, idct
import hvs as hvs
import cv2
import matplotlib.pyplot as plt


def definitive_attack(originalImage, watermarkedImage):
  #this function try first all the attacks alone, then try to combine them
  #it should be return the best wpsnr
  print("AWGN")
  awgn_attackedImage, awgn_wpsnrWatermarkAttacked, awgn_decisionMade, awgn_std = awgn_bf_best(originalImage, watermarkedImage)
  print("awgn_wpsnrWatermarkAttacked=",awgn_wpsnrWatermarkAttacked)
  print("BLUR")
  blur_attackedImage, blur_wpsnrWatermarkAttacked, blur_decisionMade, blur_sigma = blur_bf_best(originalImage, watermarkedImage)
  print("blur_wpsnrWatermarkAttacked=",blur_wpsnrWatermarkAttacked)
  print("SHARPENING")
  sharpening_attackedImage, sharpening_wpsnrWatermarkAttacked, sharpening_decisionMade, sharpening_sigma, sharpening_alpha = sharpening_bf_best(originalImage, watermarkedImage)
  print("sharpening_wpsnrWatermarkAttacked=",sharpening_wpsnrWatermarkAttacked)
  print("NoJPEG")
  #jpegcompression
  print("RESIZING")
  resizing_attackedImage, resizing_wpsnrWatermarkAttacked, resizing_decisionMade, resizing_scale = resizing_bf_best(originalImage, watermarkedImage)
  print("resizing_wpsnrWatermarkAttacked=", resizing_wpsnrWatermarkAttacked)
  print("MEDIAN")
  median_attackedImage, median_wpsnrWatermarkAttacked, median_decisionMade, median_scale = median_bf_best(originalImage, watermarkedImage)
  print("resizing_wpsnrWatermarkAttacked=",median_wpsnrWatermarkAttacked)
  return 0

number_image = input("Enter name of the image without '.bmp'":")
string_image=str(number_image)+'.bmp'
image = cv2.imread(string_image, 0)
#Embedding
watermarked = embedding(image, MARK)
definitive_attack(image, watermarked)
