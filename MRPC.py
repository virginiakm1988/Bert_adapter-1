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
from sklearn.metrics import f1_score

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
random.seed(87)
np.random.seed(87)
torch.manual_seed(87)
torch.cuda.manual_seed(87)
torch.cuda.manual_seed_all(87)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformers import BertTokenizer, BertModel

data_path = './GLUE/MRPC/train.csv'
df = pd.read_csv(data_path, sep='\t')

df.columns = ['label','sen1','sen2'] #for QQP

test_path = './GLUE/MRPC/test.csv'
df_test = pd.read_csv(test_path, sep='\t')

df_test.columns = ['label','sen1','sen2'] #for QQP

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
                df['sen1'][index],  # the sentence to be encoded
                df['sen2'][index],
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 350,  # maximum length of a sentence
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            label = df['label'][index]
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
            label = df_test['label'][index]    
              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        return input_ids.view(350), attn_mask.view(350), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

train_dataset = Allen('train')
test_dataset = Allen('test')
train_dataloader = DataLoader(train_dataset,batch_size=16, num_workers = 20)
test_dataloader = DataLoader(test_dataset,batch_size=16, num_workers = 20)

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
    plt.savefig('./result/MRPC/'+name+'.png')


# -

backbond = BertModel.from_pretrained("bert-base-uncased").to(device)
model = Model(backbond).to(device)
loss_funtion = nn.CrossEntropyLoss()
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr = lr)

from tqdm import tqdm
from time import sleep

title = 'mixed'
print(title)
print('learning rate = ', lr)
print('Start training !!!', title)
best_acc = 0
best_f1 = 0
accuracy = []
f1 = []
for epoch in range(50):
    epoch_start = time.time()
    model.train()
    for batch_id, data in enumerate(tqdm(train_dataloader)):
        tokens, mask, label = data
        tokens, mask, label = tokens.to(device),mask.to(device), label.to(device)
        output = model(tokens = tokens, mask = mask)
        loss = loss_funtion(output, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        sleep(0.001)

    epoch_finish = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0 
        my_ans = []
        real_ans = []
        for batch_id, data in enumerate(tqdm(test_dataloader)):
            tokens, mask, label = data
            tokens, mask, label = tokens.to(device),mask.to(device), label.to(device)
            output = model(tokens=tokens, mask=mask)
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
    score = f1_score(real_ans, my_ans)
    accuracy.append(correct/count)
    f1.append(score)
    if correct/count >= best_acc:
        best_acc = correct/count
        best_f1 = score
        best_epoch = epoch
    end = time.time()
    print(title)
    print('epoch = ', epoch + 1)
    print('acc = ', correct/count)
    print('f1 = ', score)
    print('time = ', epoch_finish - epoch_start)
    print('best epoch = ', best_epoch + 1)
    print('best acc = ', best_acc)
    print('best f1 = ', best_f1)
    if epoch == 0:
        print('預計train時間 = ', 50*(end-epoch_start)/60, '分鐘')
    print('=====================================')
    if epoch == 10:
        plotImage(accuracy,title)
plotImage(accuracy,title + '_acc')
plotImage(f1,title + '_f1')
print(title)
print('Done !!!')
