import pandas as pd
# Torch 
import torch.nn as nn
import torch
import torch.optim as optim
# Torch dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import math
# 
from model import HTML_NN
import copy
import time 

t_start = time.time()
BATCH_SIZE = 4
DEVICE = 'cpu' # 'cuda:0'
NUM_OF_WORKER = 0
CKPT_DIR = 'ckpt/'
CALCULATE_MEAN_STD = False
TRAIN_VAL_SPLIT = False 
torch.manual_seed(123)
N_EPOCH = 50

if CALCULATE_MEAN_STD:
    F_MEANS = []
    F_STDS = []
else:
    from normalize_parameter import F_MEANS, F_STDS

class HTML_DATA(Dataset):
    def __init__(self):
        #load data
        df = pd.read_csv('df_processed.csv')
        df_sta = pd.read_csv('data/status.csv')
        
        # Clean column
        df = df.set_index('Customer ID')
        df_sta['Churn Category'] = df_sta['Churn Category'].map({'No Churn':0,'Competitor':1,'Dissatisfaction':2,'Attitude':3,'Price':4,'Other':5})
        df_train = df.reindex(df_sta['Customer ID'])
        df_sta = df_sta.set_index('Customer ID')
        self.y = df_sta.to_numpy().ravel()
        self.X = df_train.to_numpy()
        self.X = torch.from_numpy(self.X)
        self.X = self.X.float()

        self.len = self.X.shape[0]

        # Calcuate means, std
        if CALCULATE_MEAN_STD:
            for f_idx in range(self.X.shape[1]):
                mean = 0
                std = 0
                for r_idx in range(self.X.shape[0]):
                    mean +=  self.X[r_idx][f_idx].item()/self.len
                    std  += (self.X[r_idx][f_idx].item()**2)/self.len
                F_MEANS.append(mean)
                F_STDS.append( math.sqrt( std - (mean**2) ) )
            print(f"F_MEANS = {F_MEANS}")
            print(f"F_STDS = {F_STDS}")

        # Normalizize Input data
        for r_idx in range(self.X.shape[0]):
            for f_idx in range(self.X.shape[1]):
                self.X[r_idx][f_idx] = (self.X[r_idx][f_idx].item() - F_MEANS[f_idx])/F_STDS[f_idx]
        
        self.prob = self.balance_prob()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

    def balance_prob(self):
        # Reference: https://blog.thecodingday.com/2021/01/pytorch-%E4%BD%BF%E7%94%A8sampler%E8%99%95%E7%90%86imbalanced-data/
        y = list(self.y)
        unique_label_ids = list(set(y))    
        label_probs = []
        for label_id in range(len(unique_label_ids)):
            label_id_count = y.count(label_id)
            label_probs.append(1./label_id_count)
            # print(f"{label_id} class : {label_id_count} ({100*label_id_count/len(y)}%)")
        dataset_element_weights = [] # each element prob
        for label_id in y:                
            dataset_element_weights.append(label_probs[label_id])
        return dataset_element_weights

class HTML_SPLIT_DATA(Dataset):
    def __init__(self, data_csv, target_csv):
        # load data
        df = pd.read_csv(data_csv)
        df_sta = pd.read_csv(target_csv)
        
        # Clean column
        df = df.set_index('Customer ID')
        # df_sta['Churn Category'] = df_sta['Churn Category'].map({'No Churn':0,'Competitor':1,'Dissatisfaction':2,'Attitude':3,'Price':4,'Other':5})
        df = df.reindex(df_sta['Customer ID'])
        df_sta = df_sta.set_index('Customer ID')
        self.y = df_sta.to_numpy().ravel()
        self.X = df.to_numpy()
        self.X = torch.from_numpy(self.X)
        self.X = self.X.float()

        self.len = self.X.shape[0]

        # Calcuate means, std
        if CALCULATE_MEAN_STD:
            for f_idx in range(self.X.shape[1]):
                mean = 0
                std = 0
                for r_idx in range(self.X.shape[0]):
                    mean +=  self.X[r_idx][f_idx].item()/self.len
                    std  += (self.X[r_idx][f_idx].item()**2)/self.len
                F_MEANS.append(mean)
                F_STDS.append( math.sqrt( std - (mean**2) ) )
            print(f"F_MEANS = {F_MEANS}")
            print(f"F_STDS = {F_STDS}")

        # Normalizize Input data
        for r_idx in range(self.X.shape[0]):
            for f_idx in range(self.X.shape[1]):
                self.X[r_idx][f_idx] = (self.X[r_idx][f_idx].item() - F_MEANS[f_idx])/F_STDS[f_idx]
        
        self.prob = self.balance_prob()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
    
    def balance_prob(self):
        # Reference: https://blog.thecodingday.com/2021/01/pytorch-%E4%BD%BF%E7%94%A8sampler%E8%99%95%E7%90%86imbalanced-data/
        y = list(self.y)
        unique_label_ids = list(set(y))    
        label_probs = []
        for label_id in range(len(unique_label_ids)):
            label_id_count = y.count(label_id)
            label_probs.append(1./label_id_count)
            # print(f"{label_id} class : {label_id_count} ({100*label_id_count/len(y)}%)")
        dataset_element_weights = [] # each element prob
        for label_id in y:                
            dataset_element_weights.append(label_probs[label_id])
        return dataset_element_weights

class HTML_TEST_DATA(Dataset):
    def __init__(self):
        #load data
        df = pd.read_csv('df_processed.csv')
        df = df.set_index('Customer ID')
        df_testids = pd.read_csv('data/Test_IDs.csv')
        df_test = df.reindex(df_testids['Customer ID'])
        X_test = df_test.to_numpy()
        self.X = torch.from_numpy(X_test)
        self.customer_ids = df_test.index
        self.X = self.X.float()
        self.len = self.X.shape[0]

        # Normalizize Input data
        for r_idx in range(self.X.shape[0]):
            for f_idx in range(self.X.shape[1]):
                self.X[r_idx][f_idx] = (self.X[r_idx][f_idx].item() - F_MEANS[f_idx])/F_STDS[f_idx]
                # print(self.X[r_idx][f_idx])

    def __getitem__(self, index):
        return self.customer_ids[index], self.X[index]
    
    def __len__(self):
        return self.len

def save_checkpoint(checkpoint_path, model):
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

# Load dataset
print("loading training dataset.....")
if not TRAIN_VAL_SPLIT:
    trainset = HTML_DATA()
    validset = HTML_DATA()
else:
    trainset = HTML_SPLIT_DATA("df_processed_train.csv", "data/status_train.csv")
    validset = HTML_SPLIT_DATA("df_processed_val.csv", "data/status_val.csv")
testset  = HTML_TEST_DATA()
print("Complete image loading")

sampler = WeightedRandomSampler(weights = trainset.prob, 
                                num_samples=len(trainset),
                                replacement=True)
# normal_dataloader
# trainset_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_OF_WORKER)
# Imbalance sampler
trainset_loader = DataLoader(trainset, batch_size = BATCH_SIZE, sampler = sampler,  num_workers=NUM_OF_WORKER)
validset_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=NUM_OF_WORKER)
testset_loader  = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
print(f"Cuda is available: {use_cuda}")
torch.manual_seed(123)
device = torch.device(DEVICE)
print('Device used:', device)

def train(model, epoch):
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,2,2,2,2,2]).float() )
    # criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode
    best_f1 = -1
    best_model = None
    for ep in range(epoch):
        avg_loss = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]
            avg_loss += loss.item()/len(trainset_loader)

        f1_score = valid(model)
        # train_score, val_score = crossvalid(model, criterion,optimizer,dataset=tiny_dataset)
        print(f'Train Epoch: {ep} Loss: {round(avg_loss, 6)}, f1 score: {round(f1_score, 6)}')
        if best_f1 < f1_score:
            print(f"This is current best model.")
            best_model = copy.deepcopy(model)
            best_f1 = f1_score
    save_checkpoint('best.pth', best_model)
    return best_f1, best_model

def valid(model):
    model.eval()
    tp = [0,0,0,0,0,0] # True positive
    fp = [0,0,0,0,0,0] # Fasle Positive
    fn = [0,0,0,0,0,0] # False Negative
    with torch.no_grad():
        for i, (data, target) in enumerate(validset_loader):
            data = data.to(device)
            y = model(data)
            pred = y.max(1, keepdim=True)[1] # get the index of the max log-probability
            if pred.item() == target.item(): # Debug
                tp[pred.item()] += 1
            else:
                fp[pred.item()] += 1
                fn[target.item()] += 1
        
        # Calcuate f1-score for each class
        f_score_list = []
        for c in range(6):
            try:
                precision = tp[c]/(tp[c] + fp[c])
                recall    = tp[c]/(tp[c] + fn[c])
                f_score = 2*precision*recall/(precision + recall)
            except:
                f_score = 0
            f_score_list.append(f_score)
            print(f"f_score for class {c} = {f_score}")
        # print(f"Avg f1 score: {sum(f_score_list)/len(f_score_list)}")
    return sum(f_score_list)/len(f_score_list)

def predict(model):
    model.eval()
    s = "Customer ID,Churn Category\n"
    with torch.no_grad():
        for i, (custum_id, data) in enumerate(testset_loader):
            data = data.to(device)
            y = model(data)
            pred = y.max(1, keepdim=True)[1] # get the index of the max log-probability
            s += f"{custum_id[0]},{pred.item()}\n"
    with open("torch_train.csv", 'w') as f:
        f.write(s)

model = HTML_NN().to(device)
best_f1, best_model = train(model, N_EPOCH)
predict(best_model)
print(f"best f1 score = {best_f1}")

print(f"Take total {time.time() - t_start} sec.")