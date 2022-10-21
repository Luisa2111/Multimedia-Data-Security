def blur_bf(originalImage, watermarkedImage, sigma_min, sigma_max, sigma_step, direction):
  if direction==1:
    #go this way ->
    for sigma in np.arange(sigma_min, sigma_max+sigma_step, sigma_step):
      attackedImage=blur(watermarkedImage, sigma)
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
      #I want to destroy the watermark
      print("Forward step=",sigma)
      if decisionMade==0:
        return [attackedImage, wpsnrWatermarkAttacked, decisionMade, sigma]
  elif direction==-1:
    #go the other way <-
    for sigma in np.flip(np.arange(sigma_min-sigma_step,sigma_max+sigma_step, sigma_step)):
      attackedImage=blur(watermarkedImage, sigma)
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
      #I want to find the watermark
      print("Backward step=",sigma)
      print(wpsnrWatermarkAttacked)
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
