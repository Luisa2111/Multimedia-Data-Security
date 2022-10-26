from psnr import *
from random import randint


MARK = np.load('ef26420c.npy')
mark = np.array([(-1) ** m for m in MARK])

for _ in range(100):
    print(similarity(mark, np.array([(-1)**randint(0,1) for _ in range(1024)])))