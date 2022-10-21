import embedding as em
import detection as dt
import attack as at
from psnr import similarity, wpsnr
import cv2


def main():
    # settings
    mark_size = 1024
    alpha = 0.3
    v = 'multiplicative'

    # generate a watermark (in the challenge, we will be provided a mark)
    mark = em.generate_mark(mark_size)

    # embed watermark into three different pictures of 512x512 (as it will be in the challenge)
    pictures = ['watermarking-images/lena.bmp', 'watermarking-images/baboon.bmp', 'watermarking-images/cameraman.tif']
    watermarked_pictures = []
    name_image = 'lena.bmp'
    name_out = 'wat_lena.bmp'
    watermarked = em.embedding(name_image, mark, alpha, name_output=name_out)
    mark_ex = dt.extraction(image = cv2.imread(name_image, 0), watermarked=watermarked, mark_size=mark.size,alpha=alpha)
    #print(len([ m for m in mark_ex if (m > 0.9999 and m < 0.0001) ]))
    sim = (similarity(mark_ex,mark))
    from psnr import compute_thr
    T = 16.20757257978337
    #T = compute_thr(sim,mark_size, w = mark)
    print(T)

    atk = at.random_attack(watermarked)
    for _ in range(2):
        atk = at.random_attack(atk)
    mark_atk = dt.extraction(image = cv2.imread(name_image, 0), watermarked=atk, mark_size=mark.size,alpha=alpha)
    print(mark_atk)
    print(len([m for m in mark_atk if ( m > 0)]))
    name_atk = 'atk_lena.bmp'
    cv2.imwrite(name_atk, atk)

    print(dt.detection(name_image,name_out,name_atk,mark,T,alpha))


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
