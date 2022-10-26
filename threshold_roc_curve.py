import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import embedding_sub_cap_flat1 as em
import detection_sub_cap_flat1 as dt
import attack as at
import psnr as ps


# mark_size, alpha and v are parameters for generating mark (which we do not have to do in the challenge)
# and the spread spectrum (depending on our embedding function, these might be unnecessary)

def compute_roc(mark_size, alpha, mark):
    # get all sample images
    files = []
    for root, dirs, filenames in os.walk('sample-images-roc'):
        files.extend(filenames)
    # compute scores and labels
    scores = []
    labels = []
    for i in range(0, len(files)):
        print(i, ':', files[i])
        image = cv2.imread("".join(['./sample-images-roc/', files[i]]), 0)
        watermarked = em.embedding(name_image="".join(['./sample-images-roc/', files[i]]), mark=mark, alpha=alpha)
        mark1 = dt.extraction(image, watermarked, mark_size=mark.size, alpha=alpha)
        # print("sim normal",ps.similarity(mark,mark1))
        # code added to complain with the fact that we should not use the mark
        # mark = dt.extraction(image, watermarked, alpha = alpha, mark_size=1024)
        sample = 0
        while sample < 5:  # unsure how many samples we should include in the dataset, gonna send Andrea an e-mail
            # about it
            # fakemark = np.random.standard_normal(mark_size)
            # fakemark = np.uint8(np.rint(fakemark))
            fakemark = dt.extraction(image, cv2.imread('fakemarks/wat_' + files[i].rsplit( ".", 1 )[ 0 ] + '-' + str(sample).zfill(2) + '.bmp',0), mark_size, alpha=alpha)
            res_att,i = at.random_attack(watermarked, output=True)
            w_ex = dt.extraction(image, res_att, mark_size, alpha=alpha)
            scores.append((ps.similarity(mark1, w_ex)))
            labels.append(1)
            scores.append((ps.similarity(mark1, fakemark)))
            # print('scores ',scores[-2:],'atk',i)
            labels.append(0)
            sample += 1

    # compute ROC
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    # compute AUC
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    p_value = 0.05
    idx_tpr = np.where((fpr - p_value) == min(i for i in (fpr - p_value) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])

    p_value = 0.01
    idx_tpr = np.where((fpr - p_value) == min(i for i in (fpr - p_value) if i > 0))
    print('For a FPR approximately equals to 0.01 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.01 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])

    p_value = 0.1
    idx_tpr = np.where((fpr - p_value) == min(i for i in (fpr - p_value) if i > 0))
    print('For a FPR approximately equals to 0.1 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.1 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])

    """idx_tpr = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])"""

if __name__ == "__main__":
    alpha = 10
    MARK = np.load('ef26420c.npy')
    mark = np.array([(-1) ** m for m in MARK])
    compute_roc(mark.size, alpha=alpha, mark=mark)