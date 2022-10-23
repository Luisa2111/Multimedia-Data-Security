def jpeg_compression_bf(originalImage, watermarkedImage, qf_min, qf_max, qf_step, direction):
  if direction==1:
    #go this way ->
    for qf in np.arange(qf_min, qf_max+qf_step, qf_step):
      attackedImage=jpeg_compression(watermarkedImage, qf)
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
      #I want to destroy the watermark
      if decisionMade==0:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf
  elif direction==-1:
    #go the other way <-
    for qf in np.flip(np.arange(qf_min-qf_step,qf_max+qf_step, qf_step)):
      attackedImage=jpeg_compression(watermarkedImage, qf)
      decisionMade, wpsnrWatermarkAttacked = detection(originalImage, watermarkedImage, attackedImage)
      #I want to find the watermark
      if decisionMade==1:
        return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf
  return "blur does not work"
  
def jpeg_compression_bf_best(originalImage, watermarkedImage):
  #it seems that the seed does not influence much the attack
  attackedImage, wpsnrWatermarkAttacked, decisionMade, qf = jpeg_compression_bf(originalImage, watermarkedImage, 0   , 50    , 1   , 1)
  #I'm sure that with std-1 the watermark will be present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, qf = jpeg_compression_bf(originalImage, watermarkedImage, qf-1, qf    , 0.1 , -1)
  #I'm sure that with std+0.1 the watermark will be NOT present as I want
  attackedImage, wpsnrWatermarkAttacked, decisionMade, qf = jpeg_compression_bf(originalImage, watermarkedImage, qf  , qf+0.1, 0.01, 1)
  return attackedImage, wpsnrWatermarkAttacked, decisionMade, qf
