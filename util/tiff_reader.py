import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
MAX_SIZE = (600, 600)
img = Image.open("/data4T/hzw/dataset/NAOMi/1014_2024_NAOMi_300um_data/1/neur_vol.tiff")
images = np.zeros((600,600))
for i in range(img.n_frames-290):
    img.seek(i)
    plt.imshow(img,cmap='gray')
    plt.title('frame{}'.format(i))
    plt.show()
    # images += np.array(img)
    # break

# images /= images.max()
# plt.imshow(images, cmap='gray')
# plt.show()



