def sharpening_bf(originalImage, watermarkedImage, sigma_min, sigma_max, sigma_step, alpha_min, alpha_max, alpha_step):
  listwpsnrwatermark=[]
  wpsnrValue=[]
  #Evaluation of attacks
  for sigma in np.arange(sigma_min, sigma_max+sigma_step, sigma_step):# """We have to analyse which are the best values"""
    for alpha in np.arange(alpha_min, alpha_max+alpha_step, alpha_step):
      attackedImage=sharpening(watermarkedImage, sigma, alpha) #this it the image attacked
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
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
