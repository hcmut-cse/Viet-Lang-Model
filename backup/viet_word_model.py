#!/usr/bin/env python
# coding: utf-8

# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[1]:


import os


# In[2]:


HOME_PATH = 'server'
DATA_PATH = './'

WEIGHT_PATH = os.path.join(HOME_PATH, 'weight')
if not os.path.isdir(WEIGHT_PATH):
    os.makedirs(WEIGHT_PATH)


# ## Preprocess data

# ## Load data

# In[ ]:


with open(os.path.join(DATA_PATH, 'wordlist-full.txt')) as f:
    data = f.read().split('\n')

print(len(data))


# ## Tokenize

# In[4]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


tokenizer = Tokenizer(char_level=True, filters='', split='', oov_token='*')


# In[7]:


pad_token = 0
unknow_token = 1


# In[8]:


tokenizer.fit_on_texts(data)


# In[9]:


seq_all = tokenizer.texts_to_sequences(data)


# In[10]:


len_all = [len(x) for x in seq_all]
max_len = max(len_all)


# In[11]:


seq_all = pad_sequences(seq_all, maxlen=max_len)


# In[12]:


X_all = seq_all[:,:-1]
y_all = seq_all[:,1:]


# In[13]:


X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)


# In[14]:


import torch
torch.save(tokenizer, os.path.join(DATA_PATH, 'viet_word_tokenizer.h5'))


# ## Data loader

# In[15]:


import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


# In[16]:


def build_dataset_from_tensors(X, y):
    ds = TensorDataset(X, y)
    return ds


# In[17]:


X_train = torch.tensor(X_train).long()
y_train = torch.tensor(y_train).long()
X_val = torch.tensor(X_val).long()
y_val = torch.tensor(y_val).long()


# In[18]:


train_ds = build_dataset_from_tensors(X_train, y_train)
val_ds = build_dataset_from_tensors(X_val, y_val)


# In[19]:


batch_size = 256
shuffle = True


# In[20]:


train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)


# ## Model

# In[21]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[22]:


from tqdm import tqdm


# In[23]:


class Model(nn.Module):
    def __init__(self, vocab_size, embeding_size, hidden_size):
        super(Model, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embeding_size)
        self.lstm = nn.LSTM(embeding_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        # x: BxS
        x = self.embeding(x) # BxSxE
        if hidden_state is None:
            x, hidden_state = self.lstm(x) # BxSx2H
        else:
            x, hidden_state = self.lstm(x, hidden_state) # BxSx2H
        x = F.relu(x)
        x = self.linear(x) # BxSxV
        return x, hidden_state

    def predict(self, x, hidden_state=None):
        x, hidden_state = self.forward(x, hidden_state)
        x = F.softmax(x, dim=-1) # BxSxV
        return x, hidden_state


# In[24]:


def forward_and_loss(model, x, y, loss_fn, pad_token):
    out, hidden_state = model(x)
    loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1), ignore_index=pad_token)
    return out, loss


# In[25]:


def train_model(model, optim, train_iter, loss_fn, pad_token, weight_path=None, device=None):
    total_loss = 0.0

    model.train()

    with tqdm(total=len(train_iter)) as pbar:
        for x, y in train_iter:
            if device is not None and device.type=='cuda':
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            _, loss = forward_and_loss(model, x, y, loss_fn, pad_token=pad_token)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.update(1)
            pbar.set_description("%-10s = %.6f  " % ('loss', total_loss))

    # Save model
    if weight_path is not None:
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict()
        }

        torch.save(state, weight_path)

    return total_loss


# In[26]:


def evaluate_model(model, val_iter, pad_token, device=None):
    model.eval()
    with torch.no_grad(), tqdm(total=len(val_iter)) as pbar:
        total_loss = 0.0

        for x, y in val_iter:
            if device is not None and device.type=='cuda':
                x = x.cuda()
                y = y.cuda()

            _, loss = forward_and_loss(model, x, y, F.cross_entropy, pad_token=pad_token)

            total_loss += loss.item()

            pbar.update(1)
            pbar.set_description("%-10s = %.6f  " % ('val_loss', total_loss))

    return total_loss


# ## Training

# In[27]:


vocab_size = len(tokenizer.word_index) + 1
embedding_size = 200
hidden_size = 256
learning_rate = 0.0001
loss_fn = F.cross_entropy


# In[28]:


device = torch.torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[29]:


model = Model(vocab_size, embedding_size, hidden_size)


# In[30]:


if device.type=='cuda':
    model = model.cuda()


# In[31]:


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)


# In[32]:


WEIGHT_PATH = os.path.join(HOME_PATH, 'ln_weight')
if not os.path.isdir(WEIGHT_PATH):
    os.makedirs(WEIGHT_PATH)


# In[33]:


# logging.info('\n' + '*'*50 + '\n' + "Start logging\nvocab_size=%d\nembedding_size=%d\nhidden_size=%d\nlearning_rate=%d\nbatch_size=%d\n" \
#              % (vocab_size, embedding_size, hidden_size, learning_rate, batch_size))


# In[34]:


num_epoch = 500


# In[35]:


for i in range(1, num_epoch+1):
    weight_path = None
    if i%10==0:
        weight_path = os.path.join(WEIGHT_PATH, 'epoch_%02d.h5' % i)
    print("\nEpoch %02d" % i, flush=True)
    train_loss = train_model(model, optimizer, train_dl, loss_fn, pad_token, weight_path, device)
    val_loss = evaluate_model(model, val_dl, pad_token, device)


# In[ ]:
