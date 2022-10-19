import image_processing as ip


#After embedded the image, I want to test how robusts the watermark is against the attacks
#So I am interested only in that value of std that break the watermark

def awgn_bf(image1, std_max, seed):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for std in range(std_max): """We have to analyse which are the best values"""
    image2=ip.awgn(image1, std, seed) #this it the image attacked
    wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
    if detection(image1, image2):
      listwpsnrwatermaked.append([wpsnr,std])
    else:
      listwpsnrNOwatermark.append([wpsnr,std])
  print(listwpsnrwatermarked)
  print(listwpsnrNOwatermarked)
  return listwpsnrwatermarked, listwpsnrNOwatermarked

  
def blur_bf(image1, sigma_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for sigma in range(sigma_max): """We have to analyse which are the best values"""
    image2=ip.blur(image1, sigma) #this it the image attacked
    wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
    if detection(image1, image2):
      listwpsnrwatermaked.append([wpsnr,std])
    else:
      listwpsnrNOwatermark.append([wpsnr,std])
  print(listwpsnrwatermarked)
  print(listwpsnrNOwatermarked)
  return listwpsnrwatermarked, listwpsnrNOwatermarked
  

  
def sharpening_bf(image1, sigma_max, alpha_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for sigma in range(sigma_max): """We have to analyse which are the best values"""
    for alpha in range(alpha_max): """We have to analyse which are the best values"""
      image2=ip.sharpening(image1, sigma, alpha) #this it the image attacked
      wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
      if detection(image1, image2):
       listwpsnrwatermaked.append([wpsnr,sigma,alpha])
      else:
        listwpsnrNOwatermark.append([wpsnr,sigma,alpha])
  print(listwpsnrwatermarked)
  print(listwpsnrNOwatermarked)
  return listwpsnrwatermarked, listwpsnrNOwatermarked
  

def jpeg_compression_bf(image1, qf_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for qf in range(qf_max): """We have to analyse which are the best values"""
      image2=ip.jpeg_compression(image1, qf) #this it the image attacked
      wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
      if detection(image1, image2):
       listwpsnrwatermaked.append([wpsnr,qf])
      else:
        listwpsnrNOwatermark.append([wpsnr,qf])
  print(listwpsnrwatermarked)
  print(listwpsnrNOwatermarked)
  return listwpsnrwatermarked, listwpsnrNOwatermarked


def resizing_bf(image1, scale_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for scale in range(scale_max): """We have to analyse which are the best values"""
      image2=ip.resizing(image1, scale) #this it the image attacked
      wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
      if detection(image1, image2):
       listwpsnrwatermaked.append([wpsnr,scale])
      else:
        listwpsnrNOwatermark.append([wpsnr,scale])
  print(listwpsnrwatermarked)
  print(listwpsnrNOwatermarked)
  return listwpsnrwatermarked, listwpsnrNOwatermarked


def median_bf(image1, kernel_size_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for kernel_size in range(kernel_size_max): """We have to analyse which are the best values"""
      image2=ip.median(image1, kernel_size) #this it the image attacked
      wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
      if detection(image1, image2):
       listwpsnrwatermaked.append([wpsnr,kernel_size])
      else:
        listwpsnrNOwatermark.append([wpsnr,kernel_size])
  print(listwpsnrwatermarked)
  print(listwpsnrNOwatermarked)
  return listwpsnrwatermarked, listwpsnrNOwatermarked
