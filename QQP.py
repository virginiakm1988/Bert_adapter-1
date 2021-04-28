# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import matthews_corrcoef
import numpy as np
from PIL import Image
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import logging
import os
import sys
import torch
import random
from tqdm import tqdm
from time import sleep
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

random.seed(int(sys.argv[4]))
np.random.seed(int(sys.argv[4]))
torch.manual_seed(int(sys.argv[4]))
torch.cuda.manual_seed(int(sys.argv[4]))
torch.cuda.manual_seed_all(int(sys.argv[4]))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = sys.argv[3]
# +
from transformers import BertTokenizer, BertModel
train_path = os.path.join(data_dir, 'QQP/train.tsv')
df_train = pd.read_csv(train_path, sep='\t')
df_train.columns = [1,2,3,'sen1','sen2','label']


val_path = os.path.join(data_dir, 'QQP/dev.tsv')
df_val = pd.read_csv(val_path, sep='\t')
df_val.columns = [1,2,3,'sen1','sen2','label']

test_path = os.path.join(data_dir, 'QQP/test.tsv')
df_test = pd.read_csv(test_path, sep='\t')
df_test.columns = ['id', 'sen1', 'sen2']

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# -

class Allen(Dataset): 
    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self.len = len(df_train)
        elif self.mode == 'val':
            self.len = len(df_val)
        else:
            self.len = len(df_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            encoded = tokenizer.encode_plus(
                df_train['sen1'][index],  # the sentence to be encoded
                df_train['sen2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 350,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = df_train['label'][index]
        elif self.mode == 'val':
            encoded = tokenizer.encode_plus(
                df_val['sen1'][index],  # the sentence to be encoded
                df_val['sen2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 350,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = df_val['label'][index]  
        else:
            encoded = tokenizer.encode_plus(
                df_test['sen1'][index],  # the sentence to be encoded
                df_test['sen2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 350,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = 0
              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        return input_ids.view(350), attn_mask.view(350), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)


# +
train_dataset = Allen('train')
val_dataset = Allen('val')
test_dataset = Allen('test')

train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=32)
test_dataloader = DataLoader(test_dataset,batch_size=32)


# -

class Model(nn.Module):
    def __init__(self, backbond):
        super(Model, self).__init__()
        self.backbond = backbond
        self.condition = "train"
        self.weight_lst= []
        self.param_lst = []
        #self.backbond.named_parameters()
        for name,param in self.backbond.named_parameters(): 
            #print(param)
            if 'LayerNorm' in name and 'attention' not in name:
                self.param_lst.append(param)
                continue
            elif 'adapter' in name:
                self.param_lst.append(param)
                continue
            else:
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(768,2),
        )
    def forward(self, tokens, mask, condition):
        self.condition = condition
        embedding = self.backbond(input_ids=tokens, attention_mask=mask)[1]
        answer = self.fc(embedding)
        return answer


def plotImage(G_losses, path):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Accuracy During Epoch")
    plt.plot(G_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    #plt.legend()
    plt.savefig(path)
def showweight(arr):
    print('Model alpha List')
    for i in range(int(len(arr)/2)):
        count = i * 2
        print('serial alpha = ', arr[count].item(), ' parallel alpha = ', arr[count+1].item())


# +
backbond = BertModel.from_pretrained("bert-base-uncased").to(device)
model = Model(backbond).to(device)
loss_funtion = nn.CrossEntropyLoss()
lr = 0.0001
optimizer = optim.AdamW(model.parameters(), lr = lr)
path = sys.argv[1]
model_path = os.path.join(path, 'QQP.ckpt')
pic_path = os.path.join(path, 'QQP.png')

print('Start training QQP!!!')
best_acc = 0
best_f1 = 0
best_epoch=0
accuracy = []
for epoch in range(20):
    epoch_start = time.time()

    #training
    model.train()
    correct = 0
    count = 0 
    my_ans = []
    real_ans = []
    for batch_id, data in enumerate(tqdm(train_dataloader)):
        condition = "train"
        tokens, mask, label = data
        tokens, mask, label = tokens.to(device),mask.to(device), label.to(device)
        output = model(tokens = tokens, mask = mask, condition = condition)

        loss = loss_funtion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.view(-1,2)
        pred = torch.max(output, 1)[1]
        for j in range(len(pred)):
            my_ans.append(int(pred[j]))
            real_ans.append(int(label[j]))
            if pred[j] == label[j]:
                correct+=1
            count+=1
    train_acc = correct/count  
    train_f1 = f1_score(real_ans, my_ans)
    epoch_finish = time.time()
    
    # 算更新完的eval
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0 
        my_ans = []
        real_ans = []
        for batch_id, data in enumerate(tqdm(val_dataloader)):
            tokens, mask, label = data
            tokens, mask, label = tokens.to(device),mask.to(device), label.to(device)
            output = model(tokens=tokens, mask=mask,condition="test")
            output = output.view(-1,2)
            pred = torch.max(output, 1)[1]
            for j in range(len(pred)):
                my_ans.append(int(pred[j]))
                real_ans.append(int(label[j]))
                if pred[j] == label[j]:
                    correct+=1
                count+=1

    score = correct/count
    f1 = f1_score(real_ans, my_ans)
    accuracy.append(score)
    if score >= best_acc:
        #print(model.weight_lst)
        best_acc = score
        best_f1 = f1
        best_epoch = epoch
        torch.save(model.state_dict(), model_path)
    end = time.time()
    #print('model_weight = ', model.weight_lst)
    print('epoch = ', epoch + 1)
    #print('eval_acc = ', score,', eval_f1 = ', f1, ", train score:", train_score)
    #print('time = ', epoch_finish - epoch_start)
    print('best epoch = ', best_epoch +1)
    print('best acc = ', best_acc)
    print('best f1 = ', best_f1)
    if epoch == 0:
        print('預計train時間 = ', 15*(end-epoch_start)/60, '分鐘')
    print('=====================================')
plotImage(accuracy,pic_path)

write_path = os.path.join(path, 'QQP.txt')
f = open(write_path, 'w')
f.write("Task = QQP\n")
f.write("Total epoch = " + str(epoch + 1) + '\n')
f.write("Train accuracy = " + str(train_acc) + '\n')
f.write("Train F1 = " + str(train_f1) + '\n')
f.write("Pick best epoch = " + str(best_epoch + 1) + '\n')
f.write("Pick best accuracy = " + str(best_acc) + '\n')
f.write("Pick best F1 = " + str(best_f1) + '\n')
f.close()
print('Done QQP!!!')

'''
model = Model(backbond).to(device)
ckpt = torch.load(model_path + 'QQP.ckpt')
model.load_state_dict(ckpt)
model.eval()
if model_path == './alpha_one/':
    print('QQP')
    showweight(model.weight_lst)
'''
