

#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np

f = np.load('rolls.npz')
print(f.files)
h_d = f['arr_1']
h_traj = f['arr_0']

print(h_traj.shape)
print(h_d.shape)

np.set_printoptions(precision=3, linewidth=200)

h1 = h_d[40:,20:40]
h2 = h_traj[40:,20:40]
print(h1,"\n")
# print(h_traj[99,4,:,:])
print(h2,"\n")

print(np.sum(np.square(h1-h2)),"\n")