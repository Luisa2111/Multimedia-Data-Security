import image_processing as ip


#To be implemented in order to detect the watermark
def detection(image1, image2):
  return image1-image2


#After embedded the image, I want to test how robusts the watermark is against the attacks
#So I am interested only in that value of std that break the watermark

def awgn_bf(image1, std_max, seed):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for std in range(std_max):
    image2=ip.awgn(image1, std, seed) #this it the image attacked
    wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
    if detection(image1, image2):
      listwpsnrwatermaked.append([wpsnr,std])
    else
      listwpsnrNOwatermark.append([wpsnr,std])
  print(listwpsnrwatermarked)
  print(listwpsnrNOwatermarked)

  
def blur_bf(image1, sigma_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for sigma in range(sigma_max):
    image2=ip.blur(image1, sigma) #this it the image attacked
    wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
    if detection(image1, image2):
      listwpsnrwatermaked.append([wpsnr,std])
    else
      listwpsnrNOwatermark.append([wpsnr,std])
  print(listwpsnrwatermarked)

  
def sharpening_bf(image1, sigma_max, alpha_max):
  listwpsnrwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is present
  listwpsnrNOwatermark=[] #I will save here all the values of the wpsnr after the attack was done and the watermark is not present
  for sigma in range(sigma_max):
    for alpha in range(alpha_max):
      image2=ip.blur(image1, sigma, alpha) #this it the image attacked
      wpsnr=ip.wpsnr(image1, image2) #this evaluate the wpsnr
      if detection(image1, image2):
       listwpsnrwatermaked.append([wpsnr,sigma,alpha])
      else
        listwpsnrNOwatermark.append([wpsnr,sigma,alpha])
    print(listwpsnrwatermarked)
  

def 
