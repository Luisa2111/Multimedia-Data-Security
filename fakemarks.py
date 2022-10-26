import embedding
import embeddingSS
import cv2
import numpy as np

if __name__ == "__main__":
    path_out = 'fakemarks/'
    mark_size = 1024

    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    for im in range(101):
        img = str(im).zfill(4)
        name_image = 'sample-images-roc/' + img + '.bmp'
        name_out = path_out + 'wat_' + img + '-00' + '.bmp'
        embeddingSS.embedding(name_image, mark, name_output=name_out,alpha=0.05)
        name_out = path_out + 'wat_' + img + '-01' + '.bmp'
        embeddingSS.embedding(name_image, mark, name_output=name_out)
        name_out = path_out + 'wat_' + img + '-02' + '.bmp'
        embedding.embedding(name_image, mark, name_output=name_out)
        name_out = path_out + 'wat_' + img + '-03' + '.bmp'
        embedding.embedding(name_image, mark, name_output=name_out, alpha= 0.5)
        name_out = path_out + 'wat_' + img + '-04' + '.bmp'
        embedding.embedding(name_image, mark, name_output=name_out, alpha = 1)

