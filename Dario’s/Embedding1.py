def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0

    difference = img1 - img2
    same = not np.any(difference)
    if same is True:
        return 9999999
    csf = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels

def embedding(image):
  image[0,0]=0
  return image

#1=watermarkPresent, 2=noWatermark
def detection(originalImage, watermarkedImage, attackedImage):
  wpsnrValue=wpsnr(watermarkedImage, attackedImage)
  if (watermarkedImage-attackedImage).all()==0:
     present=1
  else:
    present=0
  return present, wpsnrValue
