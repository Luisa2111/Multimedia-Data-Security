import numpy as np
from scipy.signal import convolve2d
from math import sqrt

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
  return s

def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0

  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  csf = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels

import matplotlib.pyplot as plt
def compute_thr(sim, mark_size, w):
    SIM = np.zeros(1000)
    SIM[0] = abs(sim)
    for i in range(1, 1000):
      r = np.random.uniform(0.0, 1.0, mark_size)
      SIM[i] = abs(similarity(w, r))
    plt.scatter(range(0, 1000), SIM, s=0.5)

    SIM.sort()
    t = SIM[-2]
    T = t + (0.1 * t)
    plt.hlines(T, 0, 1000, color='red')
    plt.show()
    return T