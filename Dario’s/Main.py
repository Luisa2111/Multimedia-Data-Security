import os
import cv2


if __name__ == "__main__":
  if not os.path.isfile('lena.bmp'):  
    !wget -O lena.bmp "https://drive.google.com/uc?export=download&id=17MVOgbOEOqeOwZCzXsIJMDzW5Dw9-8zm"
  if not os.path.isfile('csf.csv'):  
    !wget -O csf.csv "https://drive.google.com/uc?export=download&id=1w43k1BTfrWm6X0rqAOQhrbX6JDhIKTRW"

  im_path='lena.bmp'
  image=cv2.imread(im_path,0)

  watermarked=embedding(image)
  attackedArray=awgn_bf(image, watermarked, 3, 123)
  print(attackedArray[0][0][0])
  print(detection(image, watermarked, attackedArray[0][0][0]))

  plt.figure(figsize=(10,10))
  plt.subplot(121)
  plt.title('image')
  plt.imshow(image, cmap='gray')
  plt.subplot(122)
  plt.title('attacked')
  plt.imshow(attackedArray[0][0][0],cmap='gray')
