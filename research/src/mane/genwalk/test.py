import pyximport; pyximport.install()

import time
import random
import gen_walk

neighbors = {0: [1,2,3], 1:[0,2, 4], 2:[0, 1,3], 3:[0, 2], 4:[1]}
freq_list = [0, 0, 0, 1,1,1,2,2,2,3,3,4]
length = 10

#ret = gen_walk.random_walk(neighbors, length)
#ret = gen_walk.motif_walk(neighbors, length)
ret = gen_walk.gen_walk_fast(neighbors, freq_list, "motif_walk")
print(ret)
