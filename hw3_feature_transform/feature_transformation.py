import numpy as np
import random
Q = 2
D = 10 # Dimension of a data point

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def data_load(file_name):
    l = []
    with open(file_name, 'r') as f:
        for i in f.readlines():
            x = [float(x_n) for x_n in i.split('\n')[0].split('\t')]
            l.append(x)
    return l

def feature_transfor(data_list):
    l = []
    for i in data_list:
        x_trans = []
        for poly in range(Q+1):
            if poly == 0:
                x_trans += [1]
                continue
            x_trans += [i[j]**poly for j in range(D)]
        l.append(x_trans)
    return l

def feature_transfor_full_order(data_list):
    l = []
    for i in data_list:
        x_trans = []
        for poly in range(Q+1):
            if poly == 0:
                x_trans += [1]
                continue
            x_trans += [i[j]**poly for j in range(D)]

        # Add C10/2
        for a in range(D-1):
            for b in range(a+1):
                x_trans += [ i[a]*i[b] ]
        
        # print(len(x_trans))
        l.append(x_trans)
    return l

def feature_transfor_lower_dim(data_list, dim): 
    l = []
    for i in data_list:
        x_trans = [1]
        for d in dim:
            x_trans += [ i[d] ]
        l.append(x_trans)
    return l


def eval_error_01(y_pred, y_label):
    error_01 = 0
    for i, score in enumerate(y_pred):
        if sign(score) != y_label[i]: # This is a error prediction
            error_01 += 1/len(y_label)
    return error_01

# Get Training data
train_data = data_load('hw3_train.dat')
test_data = data_load('hw3_test.dat')

# Feature Transformation

train_data_trans = feature_transfor(train_data)
test_data_trans  = feature_transfor(test_data)

# P14
# train_data_trans = feature_transfor_full_order(train_data)
# test_data_trans  = feature_transfor_full_order(test_data)

# P15
# N_DIM_USE = 3 # From 1~10
# print(f"N_DIM_USE = {N_DIM_USE}")
# train_data_trans = feature_transfor_lower_dim(train_data, [d for d in range(N_DIM_USE)]) # dim = [0,1,2,3,4,5,6,7,8,9] means use all dimension
# test_data_trans  = feature_transfor_lower_dim(test_data, [d for d in range(N_DIM_USE)])

# P16
# dim = sorted(random.sample(range(D), 5))
# print(f"dim = {dim}")
# train_data_trans = feature_transfor_lower_dim(train_data, dim) # dim = [0,1,2,3,4,5,6,7,8,9] means use all dimension
# test_data_trans  = feature_transfor_lower_dim(test_data, dim)

# Linear regression 
X = []
Y = []
for i in range(len(train_data_trans)):
    X.append(train_data_trans[i])
    Y.append([train_data[i][-1]])
X = np.array(X)
Y = np.array(Y)
X_pesudo = np.linalg.pinv(X, rcond=1e-15, hermitian=False)
W_lin = np.matmul(X_pesudo, Y)
# print(W_lin)

# Evaluate Ein 0/1
y_pred = np.matmul(np.array(train_data_trans), W_lin)
Ein = eval_error_01(y_pred, [i[-1] for i in train_data])
print(f"Ein = {Ein}")

# Evaluate Eout 0/1
y_pred = np.matmul(np.array(test_data_trans), W_lin)
Eout = eval_error_01(y_pred, [i[-1] for i in test_data])
print(f"Eout = {Eout}")

# Print answer
print("|Ein - Eout| = {}".format(abs(Ein - Eout)))
