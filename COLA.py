# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader
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
import numpy as np
import time
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

random.seed(87)
np.random.seed(87)
torch.manual_seed(87)
torch.cuda.manual_seed(87)
torch.cuda.manual_seed_all(87)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_path = 'GLUE/CoLA/train.tsv'
df = pd.read_csv(data_path, sep='\t')

df.columns = [1,'label',3,'sen']

test_path = 'GLUE/CoLA/dev.tsv'
df_test = pd.read_csv(test_path, sep='\t')

df_test.columns = [1,'label',3,'sen']
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class Allen(Dataset): 
    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self.len = len(df)
        else:
            self.len = len(df_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            encoded = tokenizer.encode_plus(
                text=df['sen'][index],  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 100,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = df['label'][index]
        else:
            encoded = tokenizer.encode_plus(
                text=df_test['sen'][index],  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 100,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = df_test['label'][index]    
              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        return input_ids.view(100), attn_mask.view(100), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

train_dataset = Allen('train')
test_dataset = Allen('test')
train_dataloader = DataLoader(train_dataset,batch_size=256)
test_dataloader = DataLoader(test_dataset,batch_size=256)

class Model(nn.Module):
    def __init__(self, backbond):
        super(Model, self).__init__()
        self.backbond = backbond

        for name, param in self.backbond.named_parameters():  # 带有参数名的模型的各个层包含的参数遍历
            if 'LayerNorm' in name and 'attention' not in name:
                # print(name,' 沒有被freeze')
                continue
            elif 'adapter' in name:
                # print(name,' 沒有被freeze')
                continue
            else:
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 2),
        )
    def forward(self, tokens, mask):
        embedding = self.backbond(input_ids=tokens, attention_mask=mask)[1]
        answer = self.fc(embedding)
        return answer

def plotImage(G_losses, name):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Accuracy During Epoch")
    plt.plot(G_losses)
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.savefig('./result/COLA/'+name+'.png')

backbond = BertModel.from_pretrained("bert-base-uncased").to(device)
model = Model(backbond).to(device)
loss_funtion = nn.CrossEntropyLoss()
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr = lr)

title = 'mixed'
print(title)
print('learning rate = ', lr)
print('Start training !!!')
best_acc = 0
accuracy = []
train_loss = []
eval_loss = []
for epoch in range(120):
    epoch_start = time.time()
    model.train()
    for batch_id, data in enumerate(train_dataloader):
        tokens, mask, label = data
        tokens, mask, label = tokens.to(device),mask.to(device), label.to(device)
      
        output = model(tokens = tokens, mask = mask)
        loss = loss_funtion(output, label)
        if batch_id %250==0:
            train_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_finish = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0 
        my_ans = []
        real_ans = []
        for batch_id, data in enumerate(test_dataloader):
            tokens, mask, label = data
            tokens, mask, label = tokens.to(device),mask.to(device), label.to(device)
            output = model(tokens=tokens, mask=mask)
            loss = loss_funtion(output, label)
            if batch_id %250==0:
                eval_loss.append(loss.item())
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
    end = time.time()
    print('epoch = ', epoch)
    print('cor = ', score)
    print('time = ', epoch_finish - epoch_start)
    print('best epoch = ', best_epoch)
    print('best cor = ', best_acc)
    if epoch == 0:
        print('預計train時間 = ', 120*(end-epoch_start)/60, '分鐘')
    print('=====================================')
plotImage(accuracy,title)
plotImage(train_loss,'train')
plotImage(eval_loss,'eval')
print(title)
print('Done !!!')
