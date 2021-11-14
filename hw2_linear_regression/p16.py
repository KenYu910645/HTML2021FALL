import random
import numpy as np
import math

N_TRAIN = 200 # Number of training data
N_TEST = 5000 # Number of testing data
N_REPEAT = 100
# Logistic Regression
N_MAX_ITERATION = 500
RHO = 0.1
# Outliner
N_OUTLINE = 20

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def sigmoid(x):
    return math.exp(x) / (1 + math.exp(x))

Eout_lin_acc = 0
Eout_log_acc = 0
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

    # Outliner data
    for _ in range(N_OUTLINE):
        x1 = np.random.normal(6, math.sqrt(0.3), 1)[0]
        x2 = np.random.normal(0, math.sqrt(0.1), 1)[0]
        train_data.append((1, x1, x2, 1))

    # Linear regression 
    X = []
    Y = []
    for i in range(N_TRAIN + N_OUTLINE):
        X.append(train_data[i][:3])
        Y.append([train_data[i][3]])
    X = np.array(X)
    Y = np.array(Y)
    X_pesudo = np.linalg.pinv(X, rcond=1e-15, hermitian=False)
    W_lin = np.matmul(X_pesudo, Y)

    # Logistic regression
    W_log = np.array([0, 0, 0])
    for _ in range(N_MAX_ITERATION):
        acc = 0
        for i in range(N_TRAIN + N_OUTLINE):
            xn = np.array(train_data[i][:3])
            yn = train_data[i][3]
            acc += (sigmoid(-1 * yn * np.dot(W_log, xn))) * (-1 * yn * xn)
        gradient = acc / (N_TRAIN + N_OUTLINE)
        W_log = W_log - RHO * gradient

    # Calculate Linear Regression Eout (error on training dataset)
    for i in range(N_TEST):
        s = np.dot(W_lin.reshape(3), np.array(test_data[i][:3]))
        if sign(test_data[i][3]*s) != 1:
            Eout_lin_acc += 1 * (1/N_TEST)

    # Calculate Logistic Regression Eout (error on training dataset)
    for i in range(N_TEST):
        s = np.dot(W_log.reshape(3), np.array(test_data[i][:3]))
        if sign(test_data[i][3]*s) != 1:
            Eout_log_acc += 1 * (1/N_TEST)

print("(Eout_lin, Eout_log) = ({a}, {b})".format(a = str(Eout_lin_acc/N_REPEAT),
                                                 b = str(Eout_log_acc/N_REPEAT)))