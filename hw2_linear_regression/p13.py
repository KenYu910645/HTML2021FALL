import random 
import numpy as np
import math

N_TRAIN = 200 # Number of training data
N_TEST = 5000 # Number of testing data
N_REPEAT = 100

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

Ein_acc = 0
for seed in range(N_REPEAT):
    random.seed(a=seed)
    # Generate training/testing data
    train_data = [] 
    test_data = [] 
    for _ in range(N_TRAIN + N_TEST):
        y = random.randint(0, 1)
        if y == 1:
            x1 = np.random.normal(2, math.sqrt(0.6), 1)[0]
            x2 = np.random.normal(3, math.sqrt(0.6), 1)[0]
        elif y == 0: # -1 
            y = -1
            x1 = np.random.normal(0, math.sqrt(0.4), 1)[0]
            x2 = np.random.normal(4, math.sqrt(0.4), 1)[0]
        if len(train_data) < N_TRAIN:
            train_data.append((1, x1, x2, y))
        else:
            test_data.append((1, x1, x2, y))

    # Linear regression 
    X = []
    Y = []
    for i in range(N_TRAIN):
        X.append(train_data[i][:3])
        Y.append([train_data[i][3]])
    X = np.array(X)
    Y = np.array(Y)
    X_pesudo = np.linalg.pinv(X, rcond=1e-15, hermitian=False)
    W_lin = np.matmul(X_pesudo, Y)
    
    # Calculate Ein (error on training dataset)
    for i in range(N_TRAIN):
        s = np.dot(W_lin.reshape(3), np.array(train_data[i][:3]))
        Ein_acc += (Y[i]*s-1)**2 * (1/N_TRAIN)

print("Ein = " + str(Ein_acc / N_REPEAT))
