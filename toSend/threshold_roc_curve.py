import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import embedding_sub_cap_flat1 as em
import detection_sub_cap_flat1 as dt
import attack as at
import psnr as ps


def compute_roc(mark_size, alpha, mark, dim, step,
                max_splits, min_splits, sub_size
                , Xi_exp, Lambda_exp, L_exp, ceil,
                files_used = 0
                ):
    """
    the roc curve take as input all the parameters for the embedding
    files_used is necessary to decide on how many images test the roc
    """
    # get all sample images
    files = []
    for root, dirs, filenames in os.walk('sample-images-roc'):
        files.extend(filenames)
    # compute scores and labels
    scores = []
    labels = []
    if files_used <= 0:
        files_used = len(files)
    for i in range(0, files_used):
        print(i, ':', files[i])
        image = cv2.imread("".join(['./sample-images-roc/', files[i]]), 0)
        # real image embedding
        watermarked = em.embedding(name_image="".join(['./sample-images-roc/', files[i]]),
                                   mark=mark, alpha=alpha,
                                   dim = dim, step = step, max_splits=max_splits,
                                   min_splits=min_splits, sub_size=sub_size,
                                   Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil = ceil
                                   )
        print('wpsnr of ' + files[i] + ' :',ps.wpsnr(image,watermarked))
        # extraction of the mark, we use it for the comparison
        # because it was requested for the detection function,
        # and we want our roc curve to mimic a real situation
        mark1 = dt.extraction(image, watermarked, mark_size=mark.size, alpha=alpha,
                              dim = dim, step = step, max_splits=max_splits,
                              min_splits=min_splits, sub_size=sub_size,
                              Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil = ceil
                              )
        sample = 0
        while sample < 16:
            # creation of the fakemarks from random images
            # obtained by old embeddings and random attacks on the ORIGINAL image
            fakemark = dt.extraction(image, cv2.imread('fakemarks/wat_' + files[i].rsplit( ".", 1 )[ 0 ] + '-' + str(sample).zfill(2) + '.bmp',0),
                                     mark_size, alpha=alpha,
                                     dim = dim, step = step, max_splits=max_splits,
                                     min_splits=min_splits, sub_size=sub_size,
                                     Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil = ceil
                                     )
            res_att,i = at.random_attack(watermarked, output=True)
            w_ex = dt.extraction(image, res_att, mark_size,
                                 alpha=alpha, dim = dim, step = step,
                                 max_splits=max_splits,
                                 min_splits=min_splits, sub_size=sub_size,
                                 Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil = ceil
                                 )
            scores.append((ps.similarity(mark1, w_ex)))
            labels.append(1)
            scores.append((ps.similarity(mark1, fakemark)))
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
    import datetime
    e = datetime.datetime.now()
    plt.savefig("roc_out/roc_curve_%s%s%s-%s%s.png" % (e.year, e.month, e.day, e.hour, e.minute))

    # saving of the output
    import sys
    # f = open("roc_out/roc_curve_%s%s%s-%s%s.txt" % (e.year, e.month, e.day, e.hour, e.minute), "a")
    # f.close()
    sys.stdout = open("roc_out/roc_curve_%s%s%s-%s%s.txt" % (e.year, e.month, e.day, e.hour, e.minute), "a")
    print('roc curve evaluated on '+str(files_used)+' files with mode sub_cap_flat1.\nParameters :')
    print('alpha',alpha)
    print('dim',dim)
    print('step',step)
    print('max_splits', max_splits)
    print('Xi_exp',Xi_exp)
    print('Lambda_exp',Lambda_exp)
    print('L_exp', L_exp)
    print('min_splits' , min_splits)
    print('sub_size',sub_size)
    print('ceil' ,ceil)
    p_value = 0.05
    idx_tpr = np.where((fpr - p_value) == min(i for i in (fpr - p_value) if i > 0))
    print()
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

    sys.stdout.close()

    """idx_tpr = np.where((fpr - 0.05) == min(i for i in (fpr - 0.05) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])"""

    plt.show()

if __name__ == "__main__":
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
    MARK = np.load('ef26420c.npy')
    mark = np.array([(-1) ** m for m in MARK])
    compute_roc(mark.size, alpha=alpha, mark=mark,dim=dim,
                step=step, max_splits=max_splits,
                min_splits=min_splits, sub_size=sub_size,
                Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil = ceil,
                files_used= 0)
