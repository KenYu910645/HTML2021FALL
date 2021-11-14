import random
import numpy as np
from numpy import linalg as LA
import math

N = 100 # Number of trainig data

# Load training data
X = [] # Training data
Y = [] # Labels
with open('hw1_train.dat', 'r') as f:
    for i in f.readlines():
        l = [1] # X_0
        for s in i.split('\t')[:-1]:
            l.append(float(s))
        
        # Use norm to normalize 
        # X.append(np.array(l)/LA.norm(np.array(l)))
        X.append(np.array(l))
        Y.append(float(i.split('\t')[-1].split('\n')[0]))

def sign(x):
    if x > 0:
        return 1
    else:
        return -1



#------------ Random approch ------------#
# s = 0.0
# for _ in range(1000): # Do 1000 times pla
#     W = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # weight [b, w1, w2, ..., w10]
#     while True:
#         is_done = True
#         for _ in range(5*N): # Randomly find wrong point 5*N times
#             # Randomly pick a point
#             idx = random.randint(0,N-1)
#             # Update weight
#             if not ( sign(np.inner(W,X[idx])) == sign(Y[idx]) ):
#                 W = W + Y[idx]*X[idx] # *0.6211
#                 is_done = False
#                 break
#         if is_done:
#             break
#     s += LA.norm(W)**2
# print("Square norm of W = " + str(s/1000))


#------------ Iterative approch ------------#
s = 0.0
# for _ in range(1000): # Do 1000 times pla
W = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # weight [b, w1, w2, ..., w10]
import time

t = 0
while True:
    is_done = True
    # for _ in range(5*N): # Randomly find wrong point 5*N times
    for idx in range(N):
        # Randomly pick a point
        # idx = random.randint(0,N-1)
        # Update weight
        if not ( sign(np.inner(W,X[idx])) == sign(Y[idx]) ):
            # eta = 0.6211 
            # eta = (2)**(-t)
            eta = (-1*Y[idx]*np.inner(W, X[idx]))/((LA.norm(X[idx]))**2)
            # eta = (1/(1+t))
            # eta = math.floor( (-1*Y[idx]*np.inner(W, X[idx]))/((LA.norm(X[idx]))**2) + 1)
            W = W + Y[idx]*X[idx] * eta
            is_done = False
            t += 1
            time.sleep(0.5)
            break
    if is_done:
        break 
s += LA.norm(W)**2

print("Square norm of W = " + str(s/1000))