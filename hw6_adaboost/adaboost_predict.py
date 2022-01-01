import pickle
INT_MAX = 99999999999999999999
INT_MIN = -99999999999999999999

# Load GT_LIST
with open('adaboost.model', 'rb') as f:
    GT_LIST = pickle.load(f)
# print(GT_LIST)

# Load training data
train_data_x = []
train_data_y = []
with open("hw6_train.dat", 'r') as f:
    for line in f.readlines():
        train_data_x.append([float(ele) for ele in line.split("\n")[0].split(" ")[:-1] ])
        train_data_y.append(int(float(line.split("\n")[0].split(" ")[-1])))
print(f"train_data_x[0] = {train_data_x[0]}")
print(f"train_data_y[0] = {train_data_y[0]}")

# Load testing data 
test_data_x = []
test_data_y = []
with open("hw6_test.dat", 'r') as f:
    for line in f.readlines():
        test_data_x.append([ float(ele) for ele in line.split("\n")[0].split(" ")[:-1] ])
        test_data_y.append( int(float(line.split("\n")[0].split(" ")[-1])) )
print(f"test_data_x[0] = {test_data_x[0]}")
print(f"test_data_y[0] = {test_data_y[0]}")

# 
N_FEATURE = len(train_data_x[0])
N_TRAIN_DATA = len(train_data_x)
N_TEST_DATA = len(test_data_x)
print(f"Number fo features = {N_FEATURE}")
print(f"Number of train data = {N_TRAIN_DATA}")
print(f"Number of test data = {N_TEST_DATA}")

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

# p11
# Calcuate E_in (0/1 error)
E_in = 0
s, f_idx, theta, alpha_t = GT_LIST[0] # (-1, 9, 0.44824087255)
for data_i in range(N_TRAIN_DATA):
    pred = s*sign( train_data_x[data_i][f_idx] - theta )
    if pred != train_data_y[data_i]:
        E_in += 1/N_TRAIN_DATA
print(f"p11_ans = {E_in}")

# p12
E_in_max = INT_MIN
for gt in GT_LIST:
    E_in = 0
    s, f_idx, theta, alpha_t = gt
    for data_i in range(N_TRAIN_DATA):
        pred = s*sign( train_data_x[data_i][f_idx] - theta )
        if pred != train_data_y[data_i]:
            E_in += 1/N_TRAIN_DATA
    if E_in > E_in_max:
        E_in_max = E_in
print(f"p12_ans = {E_in_max}")

# p13
for t in range(len(GT_LIST)):
    E_in = 0
    for data_i in range(N_TRAIN_DATA):
        # Start Adaboost prediction 
        accumulate_vote = 0
        for gt_idx in range(t+1):
            s, f_idx, theta, alpha_t = GT_LIST[gt_idx]
            accumulate_vote += alpha_t*s*sign( train_data_x[data_i][f_idx] - theta )
        pred = sign(accumulate_vote)
        if pred != train_data_y[data_i]:
            E_in += 1/N_TRAIN_DATA
    print(f"t = {t+1}, Ein = {E_in}")
    
    # 
    if E_in <= 0.05:
        print(f"p13_ans = {t+1}")
        break


# P14
E_out = 0
s, f_idx, theta, alpha_t = GT_LIST[0] # (-1, 9, 0.44824087255)
for data_i in range(N_TEST_DATA):
    pred = s*sign( test_data_x[data_i][f_idx] - theta )
    if pred != test_data_y[data_i]:
        E_out += 1/N_TEST_DATA
print(f"p14_ans = {E_out}")

# p15
E_out = 0
for data_i in range(N_TEST_DATA):
    # Start Adaboost prediction 
    accumulate_vote = 0
    for s, f_idx, theta, alpha_t in GT_LIST:
        accumulate_vote += s*sign( test_data_x[data_i][f_idx] - theta ) # uniform
    pred = sign(accumulate_vote)
    if pred != test_data_y[data_i]:
        E_out += 1/N_TEST_DATA
print(f"p15_ans = {E_out}")

# p16
E_out = 0
for data_i in range(N_TEST_DATA):
    # Start Adaboost prediction 
    accumulate_vote = 0
    for s, f_idx, theta, alpha_t in GT_LIST:
        accumulate_vote += alpha_t*s*sign( test_data_x[data_i][f_idx] - theta ) # uniform
    pred = sign(accumulate_vote)
    if pred != test_data_y[data_i]:
        E_out += 1/N_TEST_DATA
print(f"p15_ans = {E_out}")
