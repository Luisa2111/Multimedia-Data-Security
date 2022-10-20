#After embedded the image, I want to test how robusts the watermark is against the attacks
#So I am interested only in that value of std that break the watermark
#Remember: we want that the watermark will be not present

def awgn_bf(originalImage, watermarkedImage, std_max, seed):
  listwpsnrwatermark=[]

  for std in range(std_max):# """We have to analyse which are the best values"""
    attackedImage=awgn(watermarkedImage, std, seed) #this it the image attacked
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, std])

  return listwpsnrwatermark

  
def blur_bf(originalImage, watermarkedImage, sigma_max):
  listwpsnrwatermark=[]

  for sigma in range(sigma_max):# """We have to analyse which are the best values"""
    attackedImage=blur(watermarkedImage, sigma) #this it the image attacked
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma])

  return listwpsnrwatermark
  

  
def sharpening_bf(originalImage, watermarkedImage, sigma_max, alpha_max):
  listwpsnrwatermark=[]

  for sigma in range(sigma_max):# """We have to analyse which are the best values"""
    for alpha in range(alpha_max):
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


def resizing_bf(originalImage, watermarkedImage, scale_max):
  listwpsnrwatermark=[]

  for scale in range(scale_max):# """We have to analyse which are the best values"""
    attackedImage=resizing(watermarkedImage, scale) #this it the image attacked
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, scale])
    
  return listwpsnrwatermark


def median_bf(originalImage, watermarkedImage, kernel_size_max):
  listwpsnrwatermark=[]

  for kernel_size in range(kernel_size_max):# """We have to analyse which are the best values"""
    attackedImage=median(watermarkedImage, kernel_size) #this it the image attacked
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    listwpsnrwatermark.append([attackedImage, wpsnrWatermarkAttacked, decisionMade, kernel_size])
    
  return listwpsnrwatermark
