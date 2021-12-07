import numpy as np
import random

# Output
TRAIN_OUTPUT_FILE = "hw4_train_trans.dat"
TEST_OUTPUT_FILE = "hw4_test_trans.dat"

def data_load(file_name):
    l = []
    with open(file_name, 'r') as f:
        for i in f.readlines():
            x = [float(x_n) for x_n in i.split('\n')[0].split(' ')]
            l.append(x)
    return l

def feature_transfor_full_order_3(data_list):
    ans = []
    D = len(data_list[0])
    print(f"Number of feature: {D}")

    for raw_features in data_list: # One row of data
        trans_f = []
        # 0 order polynominal
        trans_f.append(1) # x_0
        # 1 order polynominal
        for i in raw_features:
            trans_f.append(i)
        
        # 2 order polynominal
        for b_1 in range(D): # Where is the first ball 
            for b_2 in range(D): # where is the second ball
                if b_2 < b_1:
                    continue
                else:
                    trans_f.append(raw_features[b_1] * raw_features[b_2])
        
        # 3 order polynominal
        for b_1 in range(D): # Where is the first ball 
            for b_2 in range(D): # where is the second ball
                for b_3 in range(D): # where is the third ball
                    if b_2 < b_1 or b_3 < b_2:
                        continue
                    else:
                        trans_f.append(raw_features[b_1] * raw_features[b_2] * raw_features[b_3])
        ans.append(trans_f)

    # print(ans[0])
    # print(len(ans[0]))
    return ans

# Get Training data
train_data = data_load('hw4_train.dat')
print(f"Total {len(train_data)} training data loaded.")
print(f"First row of training data is {train_data[0]}")
test_data = data_load('hw4_test.dat')
print(f"Total {len(test_data)} testing data loaded.")
print(f"First row of testing data is {test_data[0]}")

# Feature Transformation
train_data_trans = feature_transfor_full_order_3( [i[:-1] for i in train_data] ) # Get rid of label
test_data_trans  = feature_transfor_full_order_3( [i[:-1] for i in test_data] ) # Get rid of label

# Convert train and test data into LIBLINEAR format
s = ""
for i, features in enumerate(train_data_trans) :
    # First index is label:
    row_s = str(int(train_data[i][-1])) + " "
    # Reset of the index is feature transformed
    for j, feature in enumerate(features):
        row_s += f"{j+1}:{feature}"
        row_s += " "
    s += row_s + "\n"
with open(TRAIN_OUTPUT_FILE, 'w') as f:
    f.write(s)
print(f"Output .dat to {TRAIN_OUTPUT_FILE}")
# print(s.split('\n')[0])


s = ""
for i, features in enumerate(test_data_trans) :
    # First index is label:
    row_s = str(int(test_data[i][-1])) + " "
    # Reset of the index is feature transformed
    for j, feature in enumerate(features):
        row_s += f"{j+1}:{feature}"
        row_s += " "
    s += row_s + "\n"
with open(TEST_OUTPUT_FILE, 'w') as f:
    f.write(s)
print(f"Output .dat to {TEST_OUTPUT_FILE}")