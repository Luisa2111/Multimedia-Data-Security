import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import embeddingSS as em
import detectionSS as dt
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
    for i in range(0, 1): #len(files)):
        print(files[i])
        image = cv2.imread("".join(['./sample-images-roc/', files[i]]), 0)
        watermarked = em.embedding(name_image="".join(['./sample-images-roc/', files[i]]), mark=mark, alpha=alpha)
        sample = 0
        while sample < 50:  # unsure how many samples we should include in the dataset, gonna send Andrea an e-mail
            # about it
            fakemark = np.random.uniform(0.0, 1.0, mark_size)
            fakemark = np.uint8(np.rint(fakemark))
            res_att = at.random_attack(watermarked)
            w_ex = dt.extraction(image, res_att, mark_size, alpha=alpha)
            scores.append(ps.similarity(mark, w_ex))
            labels.append(1)
            scores.append(ps.similarity(fakemark, w_ex))
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
    idx_tpr = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])
