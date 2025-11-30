import numpy
import numpy as np
import math
import scipy.io as scio

# f = 2.45e-3
# for i in range(30,80,1):
#     d1 = i / 10000
#     d2 = f * d1 / (d1-f)
#     if d2 / d1 <= 0.5:
#         print('d1 is {}mm and d2 is {}mm'.format(d1*1e3,d2*1e3))

sag = np.load('../data/fabrication/20241120v3_quantized_heightmap_BK7.npy')
sag = sag.flatten()
scio.savemat("/home/hzw/project/FlatScope/data/fabrication/Model20241120v3_QuantizedPhase_by_BK7.mat",
                     {'data': sag})