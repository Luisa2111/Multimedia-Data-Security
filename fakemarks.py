import embedding
import embeddingSS
import cv2
import numpy as np

if __name__ == "__main__":
    path_out = 'fakemarks/'
    mark_size = 1024

    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    name_image = 'lena.bmp'
    name_out = path_out + 'wat0_' + name_image
    embeddingSS.embedding(name_image, mark, name_output=name_out,alpha=0.05)
    name_out = path_out + 'wat1_' + name_image
    embeddingSS.embedding(name_image, mark, name_output=name_out)
    name_out = path_out + 'wat2_' + name_image
    embedding.embedding(name_image, mark, name_output=name_out)
    name_out = path_out + 'wat3_' + name_image
    embedding.embedding(name_image, mark, name_output=name_out, alpha= 0.001)
    name_out = path_out + 'wat4_' + name_image
    embedding.embedding(name_image, mark, name_output=name_out, alpha = 1)

