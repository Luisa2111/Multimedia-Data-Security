import threading



def embedding_SVD(image, alpha):
    #svd
    u, s, v = np.linalg.svd(image)

    mark = np.random.uniform(0.0, 1.0, s.size)
    mark = np.uint8(np.rint(mark))
    np.save('mark.npy', mark)
    s += alpha*mark
    watermarked = np.matrix(u) * np.diag(s) * np.matrix(v)
    return watermarked, mark

def detection_SVD(image, watermarked, alpha):
    u, s, v = np.linalg.svd(image)
    uu, s_wat, vv = np.linalg.svd(watermarked)
    w_ex = (s_wat - s)/alpha
    return w_ex

al = 10
watermarked_svd, mark_svd = embedding_SVD(image, al)
w_ex_svd = detection_SVD(image, watermarked_svd, al)

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.title('Watermarked')
plt.imshow(watermarked_svd,cmap='gray')
plt.show()

wpsnr(image,watermarked_svd)

sim = similarity(mark_svd, w_ex_svd)
T_SVD = compute_thr(sim, 512, mark_svd)