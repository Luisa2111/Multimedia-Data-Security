from embedding_ef26420c import embedding
from detection_ef26420c import detection, extraction, extraction_parallel, similarity, wpsnr
import cv2
import numpy as np
import attack as at
import matplotlib.pyplot as plt

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
threeshold = 1.87
mark_size = 1024

dir_im = 'images/'
image_name = 'lena'
name_image = dir_im + image_name + '.bmp'
name_out = dir_im + 'ef26420c_' + image_name + '.bmp'
mark = 'ef26420c.npy'
MARK = np.load('ef26420c.npy')
MARK = np.array([(-1) ** m for m in MARK])
image = cv2.imread(name_image, 0)

watermarked = embedding(name_image, name_output=name_out, name_mark= mark, alpha = alpha,
                 dim = dim, step = step, max_splits=max_splits,
                 min_splits=min_splits, sub_size=sub_size,
                 Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp , ceil = ceil)

mark_ex = extraction(image = image, watermarked=watermarked, mark_size=mark_size,alpha=alpha,
                            dim = dim, step = step, max_splits=max_splits,
                            min_splits = min_splits, sub_size = sub_size,
                            Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp , ceil = ceil)

print("wpsnr :",wpsnr(image,watermarked) )
sim = (similarity(MARK,mark_ex))
print('sim', sim)

attack_analysis = False

if attack_analysis:
    problem = set({})
    print('starting attacks')
    SIM = []
    for i in range(1, 7):
        atk = at.attack_wpsnr_fix(watermarked, i)
        mark_atk = extraction(image=image, watermarked=atk, mark_size=mark_size, alpha=alpha,
                                 dim=dim, step=step, max_splits=max_splits,
                                 min_splits=min_splits, sub_size=sub_size,
                                 Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil = ceil)
        sim = similarity(mark_ex, mark_atk)
        SIM.append(sim)
        print(i, sim, wpsnr(watermarked, atk))
        if sim < threeshold:
            problem.add(i)
    print('problems in',problem, '| min sim',min(SIM))

fakemark_analysis = False

if fakemark_analysis:
    SIM = []
    for i in range(16):
        fakemark = extraction(image, cv2.imread(
            'fakemarks_comp/wat_' + image_name + '-' + str(i).zfill(2) + '.bmp', 0),
                                 mark_size, alpha=alpha,
                                 dim=dim, step=step, max_splits=max_splits,
                                 min_splits=min_splits, sub_size=sub_size,
                                 Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil=ceil
                                 )
        sim_fake = similarity(mark_ex,fakemark)
        print('false sim',i,':',sim_fake)
        SIM.append(sim_fake)
    print(max(SIM))


roc = False
if roc:
    # get all sample images
    file = image_name
    # compute scores and labels
    scores = []
    labels = []
    sample = 0
    while sample < 16:
        # creation of the fakemarks from random images
        # obtained by old embeddings and random attacks on the ORIGINAL image
        fakemark = extraction(image, cv2.imread(
            'fakemarks_comp/wat_' + image_name + '-' + str(sample).zfill(2) + '.bmp', 0),
                                 mark_size, alpha=alpha,
                                 dim=dim, step=step, max_splits=max_splits,
                                 min_splits=min_splits, sub_size=sub_size,
                                 Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil=ceil
                                 )
        res_att = at.random_attack(watermarked, output=False)
        mark_atk = extraction(image, res_att, mark_size,
                             alpha=alpha, dim=dim, step=step,
                             max_splits=max_splits,
                             min_splits=min_splits, sub_size=sub_size,
                             Xi_exp=Xi_exp, Lambda_exp=Lambda_exp, L_exp=L_exp, ceil=ceil
                             )
        scores.append((similarity(mark_ex, mark_atk)))
        labels.append(1)
        scores.append((similarity(mark_ex, fakemark)))
        labels.append(0)
        sample += 1

    # compute ROC
    from sklearn.metrics import roc_curve, auc
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
    plt.savefig("./roc_out/roc_curve_%s%s%s-%s%s.png" % (e.year, e.month, e.day, e.hour, e.minute))

    # saving of the output
    import sys

    # f = open("roc_out/roc_curve_%s%s%s-%s%s.txt" % (e.year, e.month, e.day, e.hour, e.minute), "a")
    # f.close()
    sys.stdout = open("./roc_out/roc_curve_%s%s%s-%s%s.txt" % (e.year, e.month, e.day, e.hour, e.minute), "a")
    print('roc curve evaluated on ' + name_image + ' with mode sub_cap_flat1.\nParameters :')
    print('alpha', alpha)
    print('dim', dim)
    print('step', step)
    print('max_splits', max_splits)
    print('Xi_exp', Xi_exp)
    print('Lambda_exp', Lambda_exp)
    print('L_exp', L_exp)
    print('min_splits', min_splits)
    print('sub_size', sub_size)
    print('ceil', ceil)
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

atk = at.random_attack_param(watermarked)
name_atk = 'atk.bmp'
cv2.imwrite(name_atk,atk)
plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.title('Watermarked : ' + str(wpsnr(watermarked, image)))
plt.imshow(watermarked, cmap='gray')
plt.subplot(133)
plt.title('Attacked : ' + str(detection(name_image, name_out, name_atk)))
plt.imshow(atk, cmap='gray')
plt.show()

