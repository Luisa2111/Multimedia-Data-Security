import image_processing as ip


#After embedded the image, I want to test how robusts the watermark is against the attacks
#So I am interested only in that value of std that break the watermark

def awgn_bf(OriginalImage, WatermarkedImage, std_max, seed):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for std in range(std_max): """We have to analyse which are the best values"""
    AttackedImage=ip.awgn(WatermarkedImage, std, seed) #this it the image attacked
    wpsnrValue=ip.wpsnr(WatermarkedImage, AttackedImage) #this evaluate the wpsnr
    if detection(OriginalImage, WatermarkedImage, AttackedImage):
      listwpsnrwatermark.append([wpsnrValue,std])
    else:
      listwpsnrNOwatermark.append([wpsnrValue,std])
  print(listwpsnrwatermark)
  print(listwpsnrNOwatermark)
  return listwpsnrwatermark, listwpsnrNOwatermark

  
def blur_bf(OriginalImage, WatermarkedImage, sigma_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for sigma in range(sigma_max): """We have to analyse which are the best values"""
    AttackedImage=ip.blur(WatermarkedImage, sigma) #this it the image attacked
    wpsnrValue=ip.wpsnr(WatermarkedImage, AttackedImage) #this evaluate the wpsnr
    if detection(OriginalImage, WatermarkedImage, AttackedImage):
      listwpsnrwatermark.append([wpsnrValue,std])
    else:
      listwpsnrNOwatermark.append([wpsnrValue,std])
  print(listwpsnrwatermark)
  print(listwpsnrNOwatermark)
  return listwpsnrwatermark, listwpsnrNOwatermark
  

  
def sharpening_bf(OriginalImage, WatermarkedImage, sigma_max, alpha_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for sigma in range(sigma_max): """We have to analyse which are the best values"""
    for alpha in range(alpha_max): """We have to analyse which are the best values"""
      AttackedImage=ip.sharpening(WatermarkedImage, sigma, alpha) #this it the image attacked
      wpsnrValue=ip.wpsnr(WatermarkedImage, AttackedImage) #this evaluate the wpsnr
      if detection(WatermarkedImage, AttackedImage):
       listwpsnrwatermark.append([wpsnrValue,sigma,alpha])
      else:
        listwpsnrNOwatermark.append([wpsnrValue,sigma,alpha])
  print(listwpsnrwatermarked)
  print(listwpsnrNOwatermark)
  return listwpsnrwatermark, listwpsnrNOwatermark
  

def jpeg_compression_bf(OriginalImage, WatermarkedImage, qf_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for qf in range(qf_max): """We have to analyse which are the best values"""
      AttackedImage=ip.jpeg_compression(WatermarkedImage, qf) #this it the image attacked
      wpsnrValue=ip.wpsnr(WatermarkedImage, AttackedImage) #this evaluate the wpsnr
      if detection(WatermarkedImage, AttackedImage):
       listwpsnrwatermark.append([wpsnrValue,qf])
      else:
        listwpsnrNOwatermark.append([wpsnrValue,qf])
  print(listwpsnrwatermark)
  print(listwpsnrNOwatermark)
  return listwpsnrwatermark, listwpsnrNOwatermark


def resizing_bf(OriginalImage, WatermarkedImage, scale_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for scale in range(scale_max): """We have to analyse which are the best values"""
      AttackedImage=ip.resizing(WatermarkedImage, scale) #this it the image attacked
      wpsnrValue=ip.wpsnr(WatermarkedImage, AttackedImage) #this evaluate the wpsnr
      if detection(OriginalImage, WatermarkedImage, AttackedImage):
       listwpsnrwatermark.append([wpsnrValue,scale])
      else:
        listwpsnrNOwatermark.append([wpsnrValue,scale])
  print(listwpsnrwatermark)
  print(listwpsnrNOwatermark)
  return listwpsnrwatermark, listwpsnrNOwatermark


def median_bf(OriginalImage, WatermarkedImage, kernel_size_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for kernel_size in range(kernel_size_max): """We have to analyse which are the best values"""
      AttackedImage=ip.median(WatermarkedImage, kernel_size) #this it the image attacked
      wpsnrValue=ip.wpsnr(WatermarkedImage, AttackedImage) #this evaluate the wpsnr
      if detection(OriginalImage, WatermarkedImage, AttackedImage):
       listwpsnrwatermark.append([wpsnrValue,kernel_size])
      else:
        listwpsnrNOwatermark.append([wpsnrValue,kernel_size])
  print(listwpsnrwatermark)
  print(listwpsnrNOwatermark)
  return listwpsnrwatermark, listwpsnrNOwatermark
