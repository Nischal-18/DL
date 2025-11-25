import numpy as np

data="hello world"
chars=list(set(data))
char_to_ix={ch:i for i,ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}

vocab_size=len(chars)
hidden_size=16
seq_length=5
learning_rate=0.1

def init_weights():
    return np.random.randn(hidden_size,hidden_size+vocab_size)*0.1

Wf=init_weights()
Wi=init_weights()
Wc=init_weights()
Wo=init_weights()
Wy=np.random.randn(vocab_size,hidden_size)*0.1

bf=np.zeros((hidden_size,1))
bi=np.zeros((hidden_size,1))
bc=np.zeros((hidden_size,1))
bo=np.zeros((hidden_size,1))
by=np.zeros((vocab_size,1))

def sigmoid(x): return 1/(1+np.exp(-x))
def softmax(x): return np.exp(x)/np.sum(np.exp(x))

h_prev=np.zeros((hidden_size,1))
c_prev=np.zeros((hidden_size,1))

for step in range(500):
    p=step%(len(data)-seq_length-1)
    inputs=[char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets=[char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    x_s,f_s,i_s,c_hat_s,c_s,o_s,h_s,y_s,p_s={}, {}, {}, {}, {}, {}, {}, {}, {}
    h_s[-1]=np.copy(h_prev)
    c_s[-1]=np.copy(c_prev)
    loss=0

    for t in range(seq_length):
        x_s[t]=np.zeros((vocab_size,1))
        x_s[t][inputs[t]]=1
        z=np.vstack((h_s[t-1], x_s[t]))

        f_s[t]=sigmoid(Wf@z + bf)
        i_s[t]=sigmoid(Wi@z + bi)
        c_hat_s[t]=np.tanh(Wc@z + bc)
        c_s[t]=f_s[t]*c_s[t-1] + i_s[t]*c_hat_s[t]
        o_s[t]=sigmoid(Wo@z + bo)
        h_s[t]=o_s[t]*np.tanh(c_s[t])

        y_s[t]=Wy@h_s[t] + by
        p_s[t]=softmax(y_s[t])
        loss += -np.log(p_s[t][targets[t],0])

    if step%100==0:
        print("Loss:",loss)
