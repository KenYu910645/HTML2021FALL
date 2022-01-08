import math
from numpy import log # this is ln
import pickle
INT_MAX = 99999999999999999999
INT_MIN = -99999999999999999999

train_data_x = []
train_data_y = []
with open("hw6_train.dat", 'r') as f:
    for line in f.readlines():
        train_data_x.append([float(ele) for ele in line.split("\n")[0].split(" ")[:-1] ])
        train_data_y.append(int(float(line.split("\n")[0].split(" ")[-1])))
print(f"train_data_x[0] = {train_data_x[0]}")
print(f"train_data_y[0] = {train_data_y[0]}")


test_data_x = []
test_data_y = []
with open("hw6_test.dat", 'r') as f:
    for line in f.readlines():
        test_data_x.append([ float(ele) for ele in line.split("\n")[0].split(" ")[:-1] ])
        test_data_y.append( int(float(line.split("\n")[0].split(" ")[-1])) )
print(f"test_data_x[0] = {test_data_x[0]}")
print(f"test_data_y[0] = {test_data_y[0]}")


N_FEATURE = len(train_data_x[0])
N_TRAIN_DATA = len(train_data_x)
N_TEST_DATA = len(test_data_x)
print(f"Number fo features = {N_FEATURE}")
print(f"Number of train data = {N_TRAIN_DATA}")
print(f"Number of test data = {N_TEST_DATA}")

u = [] # importance of example
for i in range(N_TRAIN_DATA):
    u.append(1/N_TRAIN_DATA) # TODO not very sure 

# Pre-process training data
sort_feature = []
for f_idx in range(N_FEATURE):
    feature = []
    for n_idx in range(N_TRAIN_DATA):
        feature.append(train_data_x[n_idx][f_idx])
    sort_feature.append(sorted(feature))

theta_list = [] # theta_list[feature_idx][theta_idx]
for f_idx in range(N_FEATURE):
    t_list = []
    for n_idx in range(N_TRAIN_DATA):
        if n_idx == 0:
            left = INT_MIN
            right = sort_feature[f_idx][n_idx]
        else:
            left = sort_feature[f_idx][n_idx-1]
            right = sort_feature[f_idx][n_idx]
        
        # Calcuate midpoint
        t_list.append( (left + right)/2 )

    theta_list.append(t_list)
# print(theta_list)
# print(sort_feature[])

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def decision_stump():
    best = (INT_MAX, None, None, None) # [w_e, s, i, theta]
    for f_idx in range(N_FEATURE):
        for theta in theta_list[f_idx]:
            for s in (-1, 1):
                # Calcuate E_in # Weight Error 
                w_e = 0 # Weight Error 
                for data_i in range(N_TRAIN_DATA):
                    pred = s*sign( train_data_x[data_i][f_idx] - theta )
                    # If make mistake, update w_e
                    if pred != train_data_y[data_i]:
                        w_e += u[data_i] # /N_TRAIN_DATA # TODO I think this is optional
                # print(f"w_e = {w_e}")
                if w_e < best[0]: # This is current best
                    best = (w_e, s, f_idx, theta)
    return best


# Adaboost 

T = 500
GT_LIST = []
for t in range(T):
    # Get gt by decision stump
    w_e, s, i , theta = decision_stump()
    
    # calculate et
    e_t = w_e/(sum(u))
    print(f"e_t = {e_t}")
    if e_t >= 0.5:
        print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")

    # caculate dia_t(diamond_t)
    dia_t = math.sqrt((1-e_t)/e_t)
    # Update u
    for data_i in range(N_TRAIN_DATA):
        pred = s*sign( train_data_x[data_i][i] - theta )
        
        if pred == train_data_y[data_i]: # If this is correct exmaple 
            u[data_i] /= dia_t
        else:
            u[data_i] *= dia_t
    # print(f"dia_t = {dia_t}")
    # print(f"sum(u) = {sum(u)}")

    # Calcuate alpha_t
    alpha_t = log(dia_t)
    # 
    GT_LIST.append( (s, i, theta, alpha_t) )
    print(f"(s, i, theta, alpha_t) = {(s, i, theta, alpha_t)}")

    # Calculate Ein
    E_in = 0
    for data_i in range(N_TRAIN_DATA):
        accumulate_vote = 0
        for s, i, theta, alpha_t in GT_LIST:
            accumulate_vote += alpha_t*s*sign( train_data_x[data_i][f_idx] - theta )
        pred = sign(accumulate_vote)
        if pred != train_data_y[data_i]:
            E_in += 1/N_TRAIN_DATA
    print(f"E_in = {E_in}")

    # Calculate E_out
    E_out = 0
    for data_i in range(N_TEST_DATA):
        accumulate_vote = 0
        for s, i, theta, alpha_t in GT_LIST:
            accumulate_vote += alpha_t*s*sign( test_data_x[data_i][f_idx] - theta )
        pred = sign(accumulate_vote)
        if pred != test_data_y[data_i]:
            E_out += 1/N_TEST_DATA
    print(f"E_out = {E_out}")
        
print(f"GT_LIST = {GT_LIST}")

with open('adaboost.model', 'wb') as f:
    pickle.dump(GT_LIST, f)