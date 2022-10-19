import image_processing as ip

def embedding(image):
  image[0,0]=0
  return image

#1=watermarkPresent, 2=noWatermark
def detection(originalImage, watermarkedImage, attackedImage):
  wpsnrValue=ip.wpsnr(watermarkedImage, attackedImage)
  if (watermarkedImage-attackedImage).all()==0:
     present=1
  else:
    present=0
  return present, wpsnrValue
