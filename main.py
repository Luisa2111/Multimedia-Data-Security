import embedding as em
import detection as dt
import attack as at
from embedding import embedding
from detection import extraction,detection
from psnr import similarity, wpsnr
import cv2


def main():
    # settings
    mark_size = 1024
    alpha = 0.1
    v = 'multiplicative'

    # generate a watermark (in the challenge, we will be provided a mark)
    mark = em.generate_mark(mark_size)

    # embed watermark into three different pictures of 512x512 (as it will be in the challenge)
    pictures = ['watermarking-images/lena.bmp', 'watermarking-images/baboon.bmp', 'watermarking-images/cameraman.tif']
    watermarked_pictures = []
    name_image = 'lena.bmp'
    name_out = 'wat_lena.bmp'
    watermarked = embedding(name_image, mark, alpha, name_output=name_out)
    mark_ex = extraction(image = cv2.imread(name_image, 0), watermarked=watermarked, mark_size=mark.size,alpha=alpha)
    #print(mark_ex)
    sim = (similarity(mark_ex,mark))
    from psnr import compute_thr
    T = compute_thr(sim,mark_size, w = mark)
    print(T)

    atk = at.random_attack(watermarked)
    name_atk = 'atk_lena.bmp'
    cv2.imwrite(name_atk, atk)

    print(detection(name_image,name_out,name_atk,mark,T,alpha))


    #out1, w = detection(name_image,

    """
    for img_path in pictures:
        # embed mark in picture
        picture = cv2.imread(img_path, 0)
        watermarked = em.embedding(picture, img_path, mark, alpha, v)
        watermarked_pictures.append(watermarked)

        # attack picture
        attacked = at.combined_attack(picture, watermarked, img_path)

        # use detection to see whether attack was successful and mark was removed
        dt.detection(attacked, watermarked, alpha, mark_size, v)
    """
    # roccurve.compute_roc(alpha, mark_size, v)


if __name__ == "__main__":
    main()
