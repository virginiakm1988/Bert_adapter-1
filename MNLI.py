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
import random
from tqdm import tqdm
from time import sleep
import time
import csv
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
random.seed(int(sys.argv[4]))
np.random.seed(int(sys.argv[4]))
torch.manual_seed(int(sys.argv[4]))
torch.cuda.manual_seed(int(sys.argv[4]))
torch.cuda.manual_seed_all(int(sys.argv[4]))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# +
#MNLI ['index'0, 'promptID'1, 'pairID'2, 'genre'3, 'sentence1_binary_parse'4,
#       'sentence2_binary_parse'5, 'sentence1_parse'6, 'sentence2_parse'7,
#       'sentence1'8, 'sentence2'8, 'label1', 'gold_label']
# 給定前提判斷假設是否成立


data_dir = sys.argv[3]
# +
from transformers import BertTokenizer, BertModel
train_path = os.path.join(data_dir,'MNLI/train.tsv')
df_train = pd.read_csv(train_path, sep='\t',error_bad_lines=False, keep_default_na = False)


val_match_path = os.path.join(data_dir,'MNLI/dev_matched.tsv')
df_val_match = pd.read_csv(val_match_path, sep='\t',error_bad_lines=False, keep_default_na = False)

val_mismatch_path =os.path.join(data_dir,'MNLI/dev_mismatched.tsv')
df_val_mismatch = pd.read_csv(val_mismatch_path, sep='\t',error_bad_lines=False, keep_default_na = False)

test_match_path = os.path.join(data_dir,'MNLI/test_matched.tsv')
df_test_match = pd.read_csv(test_match_path, sep='\t',error_bad_lines=False, keep_default_na = False)

test_mismatch_path = os.path.join(data_dir,'MNLI/test_mismatched.tsv')
df_test_mismatch = pd.read_csv(test_mismatch_path, sep='\t',error_bad_lines=False, keep_default_na = False)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# +

class Allen(Dataset): 
    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self.len = len(df_train)
        elif self.mode == 'val_m':
            self.len = len(df_val_match)
        elif self.mode == 'val_mm':
            self.len = len(df_val_mismatch)
        elif self.mode == 'test_m':
            self.len = len(df_test_match)
        else:
            self.len = len(df_test_mismatch)

    def __getitem__(self, index):
        if self.mode == 'train':
            encoded = tokenizer.encode_plus(
                df_train['sentence1'][index],  # the sentence to be encoded
                df_train['sentence2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 1000,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            if df_train['gold_label'][index] == 'entailment':
                label = 2
            elif df_train['gold_label'][index] == 'neutral':
                label = 1
            else:
                label = 0
        elif self.mode == 'val_m':
            encoded = tokenizer.encode_plus(
                df_val_match['sentence1'][index],  # the sentence to be encoded
                df_val_match['sentence2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 1000,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            if df_val_match['gold_label'][index] == 'entailment':
                label = 2
            elif df_val_match['gold_label'][index] == 'neutral':
                label = 1
            else:
                label = 0
        elif self.mode == 'val_mm':
            encoded = tokenizer.encode_plus(
                df_val_mismatch['sentence1'][index],  # the sentence to be encoded
                df_val_mismatch['sentence2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 1000,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            if df_val_mismatch['gold_label'][index] == 'entailment':
                label = 2
            elif df_val_mismatch['gold_label'][index] == 'neutral':
                label = 1
            else:
                label = 0
        elif self.mode == 'test_m':
            encoded = tokenizer.encode_plus(
                df_test_match['sentence1'][index],  # the sentence to be encoded
                df_test_match['sentence2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 1000,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = 0
        elif self.mode == 'test_mm':
            encoded = tokenizer.encode_plus(
                df_test_mismatch['sentence1'][index],  # the sentence to be encoded
                df_test_mismatch['sentence2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 1000,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = 0
        input_ids = encoded['input_ids'][0][:128]
        attn_mask = encoded['attention_mask'][0][:128]
        return input_ids.view(128), attn_mask.view(128), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)
# +
train_dataset = Allen('train')
val_m_dataset = Allen('val_m')
val_mm_dataset = Allen('val_mm')
test_m_dataset = Allen('test_m')
test_mm_dataset = Allen('test_mm')


train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)

val_m_dataloader = DataLoader(val_m_dataset,batch_size=64)
val_mm_dataloader = DataLoader(val_mm_dataset,batch_size=64)
test_m_dataloader = DataLoader(test_m_dataset,batch_size=32)
test_mm_dataloader = DataLoader(test_mm_dataset,batch_size=32)


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
            nn.Linear(768,3),
        )
    def forward(self, tokens, mask, condition):
        self.condition = condition
        embedding = self.backbond(input_ids=tokens, attention_mask=mask)[1]
        answer = self.fc(embedding)
        return answer


# +
def plotImage(G_losses, path, match):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Corr During Epoch")
    plt.plot(G_losses)
    plt.xlabel('Epoch')
    plt.ylabel("Corr")
    #plt.legend()
    if match == 'm':
        plt.savefig(os.path.join(path, 'MNLI_m.png'))
    else:
        plt.savefig(os.path.join(path, 'MNLI_mm.png'))

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
optimizer = optim.AdamW(model.param_lst, lr = lr)
path = sys.argv[1]
model_m_path = os.path.join(path, 'MNLI_m.ckpt')
model_mm_path = os.path.join(path, 'MNLI_mm.ckpt')

print('Start training MNLI!!!')
best_acc_m = 0
best_epoch_m =0
best_acc_mm = 0
best_epoch_mm =0
accuracy_m = []
accuracy_mm = []
for epoch in range(30):
    epoch_start = time.time()
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
        output = output.view(-1,3)
        pred = torch.max(output, 1)[1]
        for j in range(len(pred)):
            if pred[j] == label[j]:
                correct+=1
            count+=1
    train_score = correct/count
    
    
    epoch_finish = time.time()
    model.eval()
    with torch.no_grad():
        correct_m = 0
        count_m = 0 
        correct_mm = 0
        count_mm = 0 
        for batch_id, data in enumerate(tqdm(val_m_dataloader)):
            tokens, mask, label = data
            tokens, mask, label = tokens.to(device),mask.to(device), label.to(device)
            output = model(tokens=tokens, mask=mask,condition="test")
            output = output.view(-1,3)
            pred = torch.max(output, 1)[1]
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    correct_m+=1
                count_m+=1
        for batch_id, data in enumerate(tqdm(val_mm_dataloader)):
            tokens, mask, label = data
            tokens, mask, label = tokens.to(device),mask.to(device), label.to(device)
            output = model(tokens=tokens, mask=mask,condition="test")
            output = output.view(-1,3)
            pred = torch.max(output, 1)[1]
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    correct_mm+=1
                count_mm+=1
                
    score_m = correct_m/count_m
    score_mm = correct_mm/count_mm
    accuracy_m.append(score_m)
    accuracy_mm.append(score_mm)
    if score_m >= best_acc_m:
        #rint(model.weight_lst)
        best_acc_m = score_m
        best_epoch_m = epoch + 1
        torch.save(model.state_dict(), model_m_path)
    if score_mm >= best_acc_mm:
        #rint(model.weight_lst)
        best_acc_mm = score_mm
        best_epoch_mm = epoch + 1
        torch.save(model.state_dict(), model_mm_path)
        
    end = time.time()
    print('epoch = ', epoch+1)
    print('best epoch match = ', best_epoch_m +1)
    print('best cor match = ', best_acc_m)
    print('best epoch mismatch = ', best_epoch_mm +1)
    print('best cor mismatch = ', best_acc_mm)
    if epoch == 0:
        print('預計train時間 = ', 30*(end-epoch_start)/60, '分鐘')
    print('=====================================')
plotImage(accuracy_m,path,'m')
plotImage(accuracy_mm,path,'mm')

write_path = os.path.join(path, 'MNLI.txt')
f = open(write_path, 'w')
f.write("Task = MNLI-m\n")
f.write("Total epoch = " + str(epoch + 1) + '\n')
f.write("Train accuracy = " + str(train_score) + '\n')
f.write("Pick best epoch = " + str(best_epoch_m + 1) + '\n')
f.write("Pick best accuracy = " + str(best_acc_m) + '\n')
f.write("Task = MNLI-mm\n")
f.write("Total epoch = " + str(epoch + 1) + '\n')
f.write("Pick best epoch = " + str(best_epoch_mm + 1) + '\n')
f.write("Pick best accuracy = " + str(best_acc_mm) + '\n')
f.close()

print('Done MNLI!!!')

'''
model = Model(backbond).to(device)
ckpt = torch.load(model_path + 'MNLI_m.ckpt')
model.load_state_dict(ckpt)
model.eval()
if model_path == './alpha_one/':
    print('MNLI_m')
    showweight(model.weight_lst)

model = Model(backbond).to(device)
ckpt = torch.load(model_path + 'MNLI_mm.ckpt')
model.load_state_dict(ckpt)
model.eval()
if model_path == './alpha_one/':
    print('MNLI_mm')
    showweight(model.weight_lst)
'''
