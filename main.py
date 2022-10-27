import numpy as np
import embedding_sub_cap_flat1 as em
import detection_sub_cap_flat1 as dt
import attack as at
from psnr import similarity, wpsnr
import cv2
import threshold_roc_curve as roc
import matplotlib.pyplot as plt

def main():
    # settings
    alpha = 5
    dim = 8
    step = 20
    max_splits = 500
    Xi_exp = 0.2
    Lambda_exp = 0.3
    L_exp = 0.2
    min_splits = 170
    sub_size = 6
    ceil = True
    threeshold = 2


    # generate a watermark (in the challenge, we will be provided a mark)
    MARK = np.load('ef26420c.npy')
    mark = np.array([(-1) ** m for m in MARK])

    # embed watermark into three different pictures of 512x512 (as it will be in the challenge)
    # name_image = 'sample-images-roc/0031.bmp'
    img = '0037'
    name_image = 'sample-images-roc/' + img +'.bmp'
    image = cv2.imread(name_image,0)
    name_out = 'watermarked.bmp'
    em.embedding(name_image, mark, alpha = alpha, name_output=name_out,
                 dim = dim, step = step, max_splits=max_splits,
                 min_splits=min_splits, sub_size=sub_size,
                 Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp , ceil = ceil)
    watermarked = cv2.imread(name_out,0)
    print("wpsnr :",wpsnr(image,watermarked) )
    mark_ex = dt.extraction(image = cv2.imread(name_image, 0), watermarked=watermarked, mark_size=mark.size,alpha=alpha,
                            dim = dim, step = step, max_splits=max_splits,
                            min_splits = min_splits, sub_size = sub_size,
                            Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp , ceil = ceil)
    # print('mark ex', (mark_ex), len(mark_ex))
    # print('mark   ', mark, len(mark))
    sim = (similarity(mark,mark_ex))
    print('sim', sim)

    problem = set({})
    print('starting attacks')
    SIM = []
    for i in range(1, 7):
        atk = at.attack_num(watermarked, i)
        mark_atk = dt.extraction(image=cv2.imread(name_image, 0), watermarked=atk, mark_size=mark.size, alpha=alpha,
                                 dim=dim, step=step, max_splits=max_splits,
                                 min_splits=min_splits, sub_size=sub_size,
                                 Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil = ceil)
        sim = similarity(mark_ex, mark_atk)
        SIM.append(sim)
        print(i, sim, wpsnr(watermarked, atk))
        if sim < 4:
            problem.add(i)
            # print(i, wpsnr(watermarked,atk))
        """print('mark atk :',mark_atk)
        print('positive >1 :', len([m for m in mark_atk if ( m > 1)]))
        print('positive 1> x > 0 :', len([m for m in mark_atk if (m > 0 and m < 1)]))
        print('negative < 0 :', len([m for m in mark_atk if (m < 0)]))
        print('similiarity :', similarity(mark,mark_atk))"""

    print('problems in',problem, '| min sim',min(SIM))


    print('analizing fake watermarked image')
    FAKE = []
    """for i in range(1,7):
        fakemark = dt.extraction(image, at.attack_num(image,i), mark_size=mark.size,alpha=alpha, step = step, max_splits=max_splits)
        FAKE.append(fakemark)
        fake_s = similarity(mark_ex,fakemark)
        print('false sim', fake_s)"""
    for i in range(16):
        fakemark = dt.extraction(image, cv2.imread('fakemarks/wat_' + img + '-' + str(i).zfill(2) + '.bmp', 0),
                                 mark.size, alpha=alpha,
                                 dim=dim, step=step, max_splits=max_splits,
                                 min_splits=min_splits, sub_size=sub_size,
                                 Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil=ceil
                                 )
        FAKE.append(similarity(mark_ex,fakemark))

    print(max(FAKE))

    """plt.hist(np.concatenate(FAKE), bins=50)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    plt.show()"""


    atk = at.attack_num(watermarked, 2)
    name_atk = 'atk.bmp'
    cv2.imwrite(name_atk, atk)
    plt.figure(figsize=(15, 6))
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.title('Watermarked : ' + str(wpsnr(watermarked, image)))
    plt.imshow(watermarked, cmap='gray')
    plt.subplot(133)
    plt.title('Attacked : ' + str(dt.detection(name_image, name_out, name_atk,
                       threeshold=threeshold,
                       mark_size=mark.size, alpha=alpha,
                       dim=dim, step=step, max_splits=max_splits,
                       min_splits=min_splits, sub_size=sub_size,
                       Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil = ceil
                       )))
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
