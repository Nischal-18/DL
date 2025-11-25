import numpy as np

text="hello world"
chars=list(set(text))
char_to_ix={ch:i for i,ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}

vocab_size=len(chars)
hidden_size=16
seq_length=5
learning_rate=0.1

Wxh=np.random.randn(hidden_size,vocab_size)*0.01
Whh=np.random.randn(hidden_size,hidden_size)*0.01
Why=np.random.randn(vocab_size,hidden_size)*0.01
bh=np.zeros((hidden_size,1))
by=np.zeros((vocab_size,1))

def softmax(x):
    e=np.exp(x-np.max(x))
    return e/e.sum()

h_prev=np.zeros((hidden_size,1))

for step in range(500):
    p=step%(len(text)-seq_length-1)
    inputs=[char_to_ix[ch] for ch in text[p:p+seq_length]]
    targets=[char_to_ix[ch] for ch in text[p+1:p+seq_length+1]]

    xs,hs,ys,ps={},{},{},{}
    hs[-1]=np.copy(h_prev)
    loss=0

    for t in range(seq_length):
        xs[t]=np.zeros((vocab_size,1))
        xs[t][inputs[t]]=1
        hs[t]=np.tanh(Wxh@xs[t] + Whh@hs[t-1] + bh)
        ys[t]=Why@hs[t] + by
        ps[t]=softmax(ys[t])
        loss += -np.log(ps[t][targets[t],0])

    dWxh=np.zeros_like(Wxh)
    dWhh=np.zeros_like(Whh)
    dWhy=np.zeros_like(Why)
    dbh=np.zeros_like(bh)
    dby=np.zeros_like(by)
    dh_next=np.zeros_like(hs[0])

    for t in reversed(range(seq_length)):
        dy=np.copy(ps[t])
        dy[targets[t]]-=1
        dWhy+=dy@hs[t].T
        dby+=dy
        dh=Why.T@dy + dh_next
        dh_raw=(1-hs[t]*hs[t]) * dh
        dbh+=dh_raw
        dWxh+=dh_raw@xs[t].T
        dWhh+=dh_raw@hs[t-1].T
        dh_next=Whh.T@dh_raw

    for dparam in [dWxh,dWhh,dWhy,dbh,dby]:
        np.clip(dparam,-5,5,out=dparam)

    for param,dparam in zip([Wxh,Whh,Why,bh,by],
                            [dWxh,dWhh,dWhy,dbh,dby]):
        param -= learning_rate*dparam

    h_prev=hs[seq_length-1]
