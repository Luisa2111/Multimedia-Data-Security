def median_bf(originalImage, watermarkedImage, scale_min, scale_max, scale_step): #Remark: scale must be odd integer
  for scale in np.arange(scale_min, scale_max+scale_step, scale_step):
    attackedImage=median_attack(watermarkedImage, scale)
    decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
    #I want to destroy the watermark
    if decisionMade==0 and wpsnrWatermarkAttacked<1000:
      return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale
  return "blur does not work"
  
def median_bf_best(originalImage, watermarkedImage):
  #it seems that the seed does not influence much the attack
  attackedImage, wpsnrWatermarkAttacked, decisionMade, scale = median_bf(originalImage, watermarkedImage, 3   ,  101   , 2)
  return attackedImage, wpsnrWatermarkAttacked, decisionMade, scale
