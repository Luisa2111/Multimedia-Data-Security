import numpy as np

import embedding_sub as em
import detection_sub as dt
import attack as at
from psnr import similarity, wpsnr
import cv2
import threshold_roc_curve as roc
import matplotlib.pyplot as plt

def main():
    # settings
    mark_size = 1024
    alpha = 10
    v = 'multiplicative'

    # generate a watermark (in the challenge, we will be provided a mark)
    MARK = np.load('ef26420c.npy')
    mark = np.array([(-1) ** m for m in MARK])
    # embed watermark into three different pictures of 512x512 (as it will be in the challenge)
    pictures = ['watermarking-images/lena.bmp', 'watermarking-images/baboon.bmp', 'watermarking-images/cameraman.tif']
    watermarked_pictures = []
    name_image = 'lena.bmp'
    image = cv2.imread(name_image,0)
    name_out = 'wat_lena.bmp'
    #em.embedding_DCT(cv2.imread(name_image,0),mark)
    watermarked = em.embedding(name_image, mark, alpha, name_output=name_out)
    mark_ex = dt.extraction(image = cv2.imread(name_image, 0), watermarked=watermarked, mark_size=mark.size,alpha=alpha)
    # np.set_printoptions(threshold=np.inf)
    print('mark ex', (mark_ex), len(mark_ex))
    print('mark   ', mark, len(mark))
    # print('els st',len([ m for m in mark_ex if (m < 1) ]))
    sim = (similarity(mark,mark_ex))
    print('sim', sim)
    from psnr import compute_thr

    # roc.compute_roc(mark.size, alpha= alpha, mark = mark)

    T = 16

    FAKE = []
    for _ in range(10):
        fakemark = dt.extraction(image, at.random_attack(image), mark_size=mark.size,alpha=alpha)
        plt.hist(fakemark, bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
        plt.show()
        fake_s = similarity(mark_ex,fakemark)
        print('false sim', fake_s)




    def compute_thr(sim, mark_size, w):
        SIM = np.zeros(1000)
        SIM[0] = abs(sim)
        for i in range(1, 1000):
            r = np.random.uniform(-1.0, 1.0, mark_size)
            SIM[i] = (similarity(w,r))
            plt.scatter(range(0, 1000), SIM, s=0.5)
        SIM.sort()
        t = SIM[-2]
        T = t + (0.1 * t)
        plt.hlines(T, 0, 1000, color='red')
        plt.show()
        return T
    #T = compute_thr(sim,mark_size, w = mark)
    print(T)
    problem = set({})
    print('starting attacks')
    for i in range(1,7):
        atk = at.attack_num(watermarked,i)
        #for _ in range(1):
        #    atk = at.random_attack(atk)
        mark_atk = dt.extraction(image = cv2.imread(name_image, 0), watermarked=atk, mark_size=mark.size,alpha=alpha)
        sim = similarity(mark_ex,mark_atk)
        print(i, sim, wpsnr(watermarked, atk))
        if sim < 10 :
            problem.add(i)
            # print(i, wpsnr(watermarked,atk))
        """print('mark atk :',mark_atk)
        print('positive >1 :', len([m for m in mark_atk if ( m > 1)]))
        print('positive 1> x > 0 :', len([m for m in mark_atk if (m > 0 and m < 1)]))
        print('negative < 0 :', len([m for m in mark_atk if (m < 0)]))
        print('similiarity :', similarity(mark,mark_atk))"""

    # print(problem)
    from collections import Counter
    print((problem))


    name_atk = 'atk_lena.bmp'
    cv2.imwrite(name_atk, atk)
    # print(dt.detection(name_image,name_out,name_atk,mark,T,alpha))
    plt.figure(figsize=(15, 6))
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.title('Watermarked : ' + str(wpsnr(watermarked, image)))
    plt.imshow(watermarked, cmap='gray')
    plt.subplot(133)
    plt.title('Attacked : ' + str(wpsnr(watermarked, atk)))
    plt.imshow(atk, cmap='gray')
    plt.show()

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
