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
from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np
from PIL import Image
import sys
import pandas as pd
import random
from tqdm import tqdm
from time import sleep
import time
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import csv
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
data_dir = sys.argv[3]

train_path = os.path.join(data_dir, 'RTE/train.tsv')
df_train = pd.read_csv(train_path, sep='\t',error_bad_lines=False)
df_train.columns = [1,'sen1','sen2','label']


val_path = os.path.join(data_dir, 'RTE/dev.tsv')
df_val = pd.read_csv(val_path, sep='\t',error_bad_lines=False)
df_val.columns = [1,'sen1','sen2','label']


test_path = os.path.join(data_dir, 'RTE/test.tsv')
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
            if df_train['label'][index] == 'entailment':
                label = 1
            else:
                label = 0
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
            if df_val['label'][index] == 'entailment':
                label = 1
            else:
                label = 0  
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
        token_type_ids = encoded['token_type_ids']
        return input_ids.view(350), attn_mask.view(350), token_type_ids.view(350), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)


# +
train_dataset = Allen('train')
val_dataset = Allen('val')
test_dataset = Allen('test')

train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=32)
test_dataloader = DataLoader(test_dataset,batch_size=64)

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
                elif 'fix' in sys.argv[1] and 'vector' in name:
                    print('大哥好，您把vector fix住了哦！！！')
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
def plotImage(G_losses, path):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Accuracy During Epoch")
    plt.plot(G_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(path)


# +
backbond = BertModel.from_pretrained("bert-base-uncased").to(device)
model = Model(backbond).to(device)
loss_funtion = nn.CrossEntropyLoss()
lr = 0.0001

optimizer_weight = optim.AdamW(model.weight_lst, lr = lr)
optimizer_bias = optim.AdamW(model.param_lst, lr = lr, weight_decay=0)

path = sys.argv[1]
model_path = os.path.join(path, 'RTE.ckpt')
pic_path = os.path.join(path, 'RTE.png')

print('Start training RTE!!!')
best_acc = 0
best_epoch=0
accuracy = []
for epoch in range(40): #50
    epoch_start = time.time()
    
    #training
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
        optimizer_bias.zero_grad()
        optimizer_weight.zero_grad()
        loss.backward()
        optimizer_bias.step()
        optimizer_weight.step()
        output = output.view(-1,2)
        pred = torch.max(output, 1)[1]
        for j in range(len(pred)):
            if pred[j] == label[j]:
                correct+=1
            count+=1
    train_score = correct/count  

    epoch_finish = time.time()
    # 算更新完的eval
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0 
        for batch_id, data in enumerate(tqdm(val_dataloader)):
            tokens, mask, type_id, label = data
            tokens, mask, type_id, label = tokens.to(device),mask.to(device), type_id.to(device), label.to(device)
            output = model(tokens = tokens, mask = mask, type_id = type_id)
            output = output.view(-1,2)
            pred = torch.max(output, 1)[1]
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    correct+=1
                count+=1
    score = correct/count
    accuracy.append(score)
    if score >= best_acc:
        best_acc = score
        best_epoch = epoch
        torch.save(model.state_dict(), 'RTE.ckpt')
    end = time.time()

    print('epoch = ', epoch+1)
    print('val score: ', score," train score:", train_score)
    print('best epoch = ', best_epoch+1)
    print('best acc = ', best_acc)
    if epoch == 0:
        print('預計train時間 = ', 40*(end-epoch_start)/60, '分鐘')
    print('=====================================')
#plotImage(accuracy,pic_path)
'''
write_path = os.path.join(path, 'RTE.txt')
f = open(write_path, 'w')
f.write("Task = RTE\n")
f.write("Total epoch = " + str(epoch + 1) + '\n')
f.write("Train accuracy = " + str(train_score) + '\n')
f.write("Pick best epoch = " + str(best_epoch + 1) + '\n')
f.write("Pick best accuracy = " + str(best_acc) + '\n')
f.close()
'''
print('Done RTE!!!')



# +
backbond = BertModel.from_pretrained("bert-base-uncased").to(device)
model = Model(backbond).to(device)

print('Start predict RTE!!!')

ckpt = torch.load('RTE.ckpt')
model.load_state_dict(ckpt)
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
            
#output_path = sys.argv[1]
output_path = 'gdrive/My Drive/bert'
output_file = os.path.join(output_path, path, 'RTE.tsv')
            
with open(output_file, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['Id', 'Label'])
    for idx, label in enumerate(ans):
        if label == 1:
            tsv_writer.writerow([idx, 'entailment'])
        else:
            tsv_writer.writerow([idx, 'not_entailment'])

