import torch
import torch.nn as nn
import torch.nn.functional as F
import random

data="deep learning"
chars=sorted(set(data))
vocab_size=len(chars)
char2idx={ch:i for i,ch in enumerate(chars)}
idx2char={i:ch for ch,i in char2idx.items()}

input_seq=[char2idx[c] for c in data]

def one_hot(i):
    return F.one_hot(torch.tensor(i),vocab_size).float()

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRUCell(vocab_size,128)
        self.fc=nn.Linear(128,vocab_size)

    def forward(self,x,h):
        h=self.gru(x,h)
        return self.fc(h),h

def sample(model,start,length=10):
    h=torch.zeros(128)
    idx=char2idx[start]
    result=start
    for _ in range(length-1):
        x=one_hot(idx)
        out,h=model(x,h)
        idx=torch.multinomial(F.softmax(out,dim=0),1).item()
        result += idx2char[idx]
    return result

model=GRU()
opt=torch.optim.Adam(model.parameters(),lr=0.01)
loss_fn=nn.CrossEntropyLoss()

for i in range(1001):
    h=torch.zeros(128)
    loss=0
    for t in range(len(data)-1):
        x=one_hot(input_seq[t])
        target=torch.tensor(input_seq[t+1])
        out,h=model(x,h)
        loss += loss_fn(out.unsqueeze(0), target.unsqueeze(0))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i%100==0:
        print("Loss:",loss.item())
        print("Sample:", sample(model, random.choice(data)))
