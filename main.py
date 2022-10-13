import embedding as em
import detection as dt
import attack as at
import cv2


def main():
    # settings
    mark_size = 128
    alpha = 24
    v = 'additive'

    # generate a watermark (in the challenge, we will be provided a mark)
    mark = em.generate_mark(mark_size)

    # embed watermark into three different pictures of 512x512 (as it will be in the challenge)
    pictures = ['lena.bmp', 'baboon.bmp', 'cameraman.tif']
    watermarked_pictures = []

    for img_path in pictures:
        # embed mark in picture
        picture = cv2.imread(img_path, 0)
        watermarked = em.embedding(picture, img_path, mark, alpha, v)
        watermarked_pictures.append(watermarked)

        # attack picture
        attacked = at.combined_attack(picture, watermarked, img_path)

        # use detection to see whether attack was successful and mark was removed
        dt.detection(attacked, watermarked, alpha, mark_size, v)


if __name__ == "__main__":
    main()
