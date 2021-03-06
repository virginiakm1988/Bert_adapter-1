# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from .transformers import BertTokenizer, BertModel
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from .modelconfig import get_args
args= get_args()

SEED = args.seed
data_dir = args.GLUE_path
output_path = args.output_path
model_path = os.path.join(output_path, 'model')
pred_path = os.path.join(output_path, 'result')
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(pred_path):
    os.makedirs(pred_path)
max_len = args.cola_len
batch_size = args.cola_batch
lr = args.cola_lr

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# +
train_path = os.path.join(data_dir,'CoLA/train.tsv')
df_train = pd.read_csv(train_path, sep='\t')
df_train.columns = [1,'label',3,'sen']


val_path = os.path.join(data_dir,'CoLA/dev.tsv')
df_val = pd.read_csv(val_path, sep='\t')
df_val.columns = [1,'label',3,'sen']

test_path = os.path.join(data_dir,'CoLA/test.tsv')
df_test = pd.read_csv(test_path, sep='\t')
df_test.columns = ['id', 'sen']


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
                text=df_train['sen'][index],  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = max_len,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = df_train['label'][index]
        elif self.mode == 'val':
            encoded = tokenizer.encode_plus(
                text=df_val['sen'][index],  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = max_len,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = df_val['label'][index]    
        else:
            encoded = tokenizer.encode_plus(
                text=df_test['sen'][index],  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = max_len,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = 0
              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        return input_ids.view(max_len), attn_mask.view(max_len), token_type_ids.view(max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

# +
train_dataset = Allen('train')
val_dataset = Allen('val')
test_dataset = Allen('test')

train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size)


# -

class Model(nn.Module):
    def __init__(self, backbond):
        super(Model, self).__init__()
        self.backbond = backbond
        self.weight_lst= []
        self.param_lst = []

        for name,param in self.backbond.named_parameters(): 
            if 'LayerNorm' in name and 'attention' not in name:
                self.param_lst.append(param)
                continue
            elif 'adapter' in name:
                if 'bias' in name:
                    self.param_lst.append(param)
                elif 'fix' in output_path and 'vector' in name:
                    print('??????????????????vector fix??????????????????')
                    param.requires_grad = False                    
                else:
                    self.weight_lst.append(param)
                continue
            else:
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768,2),
        )

        for name,param in self.fc.named_parameters(): 
            self.weight_lst.append(param)

    def forward(self, tokens, mask, type_id):
        embedding = self.backbond(input_ids=tokens, attention_mask=mask, token_type_ids = type_id)[1]
        answer = self.fc(embedding)
        return answer


# +
backbond = BertModel.from_pretrained("bert-base-uncased").to(device)
model = Model(backbond).to(device)
loss_funtion = nn.CrossEntropyLoss()

optimizer_weight = optim.AdamW(model.weight_lst, lr = lr)
optimizer_bias = optim.AdamW(model.param_lst, lr = lr, weight_decay=0)


print('Start training COLA!!!')
best_acc = 0
best_epoch=0
accuracy = []
for epoch in range(args.cola_epoch):
    epoch_start = time.time()
    model.train()
    correct = 0
    count = 0 
    my_ans = []
    real_ans = []
    for batch_id, data in enumerate(tqdm(train_dataloader)):

        tokens, mask, type_id, label = data
        tokens, mask, type_id, label = tokens.to(device),mask.to(device), type_id.to(device), label.to(device)
        output = model(tokens = tokens, mask = mask, type_id = type_id)

        loss = loss_funtion(output, label)
        optimizer_weight.zero_grad()
        optimizer_bias.zero_grad()
        loss.backward()
        optimizer_weight.step()
        optimizer_bias.step()
        output = output.view(-1,2)
        pred = torch.max(output, 1)[1]
        for j in range(len(pred)):
                    if label[j] == 0:
                        label[j] = -1
                    if pred[j] == 0:
                        pred[j] = -1
                    my_ans.append(int(pred[j]))
                    real_ans.append(int(label[j]))
                    if pred[j] == label[j]:
                        correct+=1
                    count+=1
    train_score = matthews_corrcoef(real_ans, my_ans)   
    
    
    epoch_finish = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0 
        my_ans = []
        real_ans = []
        for batch_id, data in enumerate(tqdm(val_dataloader)):
            tokens, mask, type_id, label = data
            tokens, mask, type_id, label = tokens.to(device),mask.to(device), type_id.to(device), label.to(device)
            output = model(tokens = tokens, mask = mask, type_id = type_id)
            output = output.view(-1,2)
            pred = torch.max(output, 1)[1]
            for j in range(len(pred)):
                if label[j] == 0:
                    label[j] = -1
                if pred[j] == 0:
                    pred[j] = -1
                my_ans.append(int(pred[j]))
                real_ans.append(int(label[j]))
                if pred[j] == label[j]:
                    correct+=1
                count+=1
    score = matthews_corrcoef(real_ans, my_ans)
    accuracy.append(score)
    if score >= best_acc:
        best_acc = score
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(model_path, 'COLA.ckpt'))
    end = time.time()
    
    print('epoch = ', epoch+1)
    print('val corr = ', score, ' train corr = ', train_score)
    print('best epoch = ', best_epoch+1)
    print('best cor = ', best_acc)
    if epoch == 0:
        print('??????train?????? = ', args.cola_epoch*(end-epoch_start)/60, '??????')
    print('=====================================')

print('Start predict COLA!!!')

model = Model(backbond).to(device)
ckpt = torch.load(os.path.join(model_path, 'COLA.ckpt'))
model.load_state_dict(ckpt)

# +
model.eval()
ans = []
with torch.no_grad():
    for batch_id, data in enumerate(tqdm(test_dataloader)):
        tokens, mask, type_id, _ = data
        tokens, mask, type_id = tokens.to(device),mask.to(device), type_id.to(device)
        output = model(tokens = tokens, mask = mask, type_id = type_id)
        output = output.view(-1,2)
        pred = torch.max(output, 1)[1]
        for i in range(len(pred)):
            ans.append(int(pred[i]))
            
with open(os.path.join(pred_path, 'CoLA.tsv'), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['Id', 'Label'])
    for idx, label in enumerate(ans):
        tsv_writer.writerow([idx, label])