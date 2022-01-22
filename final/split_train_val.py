'''
Split training dataset into 8:2 = train:val
'''

import pandas as pd

TRAIN_SET_RATIO = 0.8 # 

# Load csv
df     = pd.read_csv('df_processed.csv')
df_sta = pd.read_csv('data/status.csv')
# 
df = df.set_index('Customer ID')
df_sta['Churn Category'] = df_sta['Churn Category'].map({'No Churn':0,'Competitor':1,'Dissatisfaction':2,'Attitude':3,'Price':4,'Other':5})
df = df.reindex(df_sta['Customer ID'])
df_sta = df_sta.set_index('Customer ID')

# 
y = list(df_sta.to_numpy().ravel())
unique_label_ids = list(set(y))
c_labels_train = []
for label_id in range(len(unique_label_ids)):
    label_id_count = y.count(label_id)
    # print(f"{label_id} class : {label_id_count} ({100*label_id_count/len(y)}%)")
    c_labels_train.append(int(label_id_count*TRAIN_SET_RATIO))
#
df_sta = df_sta.sample(frac=1)
# df_sta = df_sta.sort_values(by=['Churn Category'])

# Get train set id
train_set_id = []
val_set_id = []
for i, row in df_sta.iterrows():
    c_id = row.name
    label = row['Churn Category']

    if c_labels_train[label] > 0: # Still have data from train
        train_set_id.append(c_id)
        c_labels_train[label] -= 1
    else:
        val_set_id.append(c_id)
print(f"Size of training set : {len(train_set_id)}")
print(f"Size of validation set : {len(val_set_id)}")

# Output status.csv and df_procesed
df_val = pd.DataFrame()
df_val.index.name = 'Customer ID'
#
df_sta_val = pd.DataFrame()
df_sta_val.index.name = 'Customer ID'

for custom_id in val_set_id:
    # Status.csv
    df_sta_val = df_sta_val.append(df_sta.loc[custom_id])
    df_sta = df_sta.drop([custom_id])
    # df.csv
    df_val = df_val.append(df.loc[custom_id])
    df = df.drop([custom_id])

# Status.csv
df_sta.to_csv("data/status_train.csv")
df_sta_val = df_sta_val.astype({"Churn Category":int})
df_sta_val.to_csv("data/status_val.csv")
# df.csv
df.to_csv("df_processed_train.csv")
df_val.to_csv("df_processed_val.csv")