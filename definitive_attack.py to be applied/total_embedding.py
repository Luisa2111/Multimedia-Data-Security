import embedding_ef26420c as emb
import psnr
import time
import cv2

for i in range(101):
	print("{num:04d}.bmp".format(num=i))
	name_image="./sample_images_roc/{num:04d}.bmp".format(num=i)
	start = time.time()
	embedded=emb.embedding(name_image, "./ef26420c.npy", alpha=5, name_output='watermarked.bmp', dim=8, step=20,
						   max_splits=500, min_splits=170, sub_size=6, Xi_exp = 0.2, Lambda_exp = 0.3, L_exp = 0.2, ceil = True)
	print('time consumed: ', time.time() - start)
	original=cv2.imread(name_image, 0)
	print("wpsrn of ", i,"th image: ", psnr.wpsnr(embedded, original), "\n")
	name_embedded="{num:04d}.bmp".format(num=i)
	cv2.imwrite("./embedded_images/" + name_embedded, embedded)
