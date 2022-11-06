import oldembeddings.embedding
import oldembeddings.embeddingSS
import cv2
import numpy as np
import attack as at

if __name__ == "__main__":
    path_out = 'toSend/fakemarks_comp/'
    mark_size = 1024

    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    img = 'lena'
    name_image = 'toSend/images/' + img + '.bmp'
    name_out = path_out + 'wat_' + img + '-00' + '.bmp'
    oldembeddings.embeddingSS.embedding(name_image, mark, name_output=name_out,alpha=0.05)
    name_out = path_out + 'wat_' + img + '-01' + '.bmp'
    oldembeddings.embeddingSS.embedding(name_image, mark, name_output=name_out)
    name_out = path_out + 'wat_' + img + '-02' + '.bmp'
    oldembeddings.embedding.embedding(name_image, mark, name_output=name_out)
    name_out = path_out + 'wat_' + img + '-03' + '.bmp'
    oldembeddings.embedding.embedding(name_image, mark, name_output=name_out, alpha= 0.5)
    name_out = path_out + 'wat_' + img + '-04' + '.bmp'
    oldembeddings.embedding.embedding(name_image, mark, name_output=name_out, alpha = 1)
    name_out = path_out + 'wat_' + img + '-05' + '.bmp'
    oldembeddings.embeddingSS.embedding(name_image, mark, name_output=name_out, alpha=0.5)
    name_out = path_out + 'wat_' + img + '-06' + '.bmp'
    oldembeddings.embeddingSS.embedding(name_image, mark, name_output=name_out, alpha= 1)
    name_out = path_out + 'wat_' + img + '-07' + '.bmp'
    oldembeddings.embedding.embedding(name_image, mark, name_output=name_out, alpha= 0.2)
    name_out = path_out + 'wat_' + img + '-08' + '.bmp'
    oldembeddings.embedding.embedding(name_image, mark, name_output=name_out, alpha=0.7)
    name_out = path_out + 'wat_' + img + '-09' + '.bmp'
    oldembeddings.embedding.embedding(name_image, mark, name_output=name_out, alpha=1.4)

    image = cv2.imread(name_image,0)
    for i in range(1,7):
        name_out = path_out + 'wat_' + img + '-' + str(i+9).zfill(2) + '.bmp'
        cv2.imwrite(name_out,at.attack_num(image,i))



