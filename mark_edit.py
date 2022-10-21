import numpy as np
from embedding import generate_mark
import matplotlib.pyplot as plt

np.random.seed(seed=42)

mark = generate_mark(1024)

logo_mark = [ mark[i:i+31] for i in range(0,1024,32)]

plt.figure(figsize=(15, 6))
plt.title('Original')
plt.imshow(logo_mark, cmap='gray')
plt.show()

