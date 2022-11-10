import numpy as np
import cokriging as ck



x = np.array([[-3,  6,  1,  5,  0, 41],
              [-8, -5,  0, 52, 38, np.nan],
              [ 3, -3,  3, 67,  6, 58]])

x0 = np.array([[  0,  0,  0],
               [ 10, 20, 33]])

block = np.array([5, 10, 5])[np.newaxis, :]
nd = np.array([3, 3, 2])
model = np.array([[ 1,  1,  1,  1,  0,  0,  0],
                  [ 4, 50, 30, 10,  0,  0, 30]])
c = np.array([[[ 20,  10,  5],
               [ 10,  25,  3],
               [  5,   3, 12]],
                
              [[ 50, -20, 15],
               [-20,  25, -7],
               [ 15,  -7, 15]],])
itype = 3
avg = np.array([0, 0, 0])
ival = 0
nk = 3
rad = 100
ntok = 2

x0s, s, sv, id, b = ck.cokri(x, x0, model, c, itype, avg, block, nd, ival, nk, rad, ntok)