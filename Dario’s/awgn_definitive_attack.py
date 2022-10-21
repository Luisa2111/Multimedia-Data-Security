def awgn_bf(originalImage, watermarkedImage, std_min, std_max, std_step, seed, direction):

  if direction==1:
    #go this way ->
    for std in np.arange(std_min, std_max+std_step, std_step):
      attackedImage=awgn(watermarkedImage, std, seed)
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
      #I want to destroy the watermark
      print("Forward step=",std)
      if decisionMade==0:
        return [attackedImage, wpsnrWatermarkAttacked, decisionMade, std]

  elif direction==-1:
    #go the other way <-
    for std in np.flip(np.arange(std_min-std_step,std_max+std_step, std_step)):
      attackedImage=awgn(watermarkedImage, std, seed)
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
      #I want to find the watermark
      print("Backward step=",std)
      if decisionMade==1:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, std

  return "awgn does not work"

def awgn_bf_best(originalImage, watermarkedImage):
  #it seems that the seed does not influence much the attack
  attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(originalImage, watermarkedImage, 0    , 50     , 1   , 123, 1)
  #I'm sure that with std-1 the watermark will be present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(originalImage, watermarkedImage, std-1, std    , 0.1 , 123, -1)
  #I'm sure that with std+0.1 the watermark will be NOT present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, std = awgn_bf(originalImage, watermarkedImage, std  , std+0.1, 0.01, 123, 1)
  return attackedImage, wpsnrWatermarkAttacked, decisionMade, std
