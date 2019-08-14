import os

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def get_model(vocab_size):
    embedding_size = 200
    hidden_size = 256

    model = Model(vocab_size, embedding_size, hidden_size)
    return model

def load_model(model, weight_path):
    if os.path.isfile(weight_path):
        state = torch.load(weight_path)
        model.load_state_dict(state)
    else:
        raise Exception("Invalid weight path")

def load_vocab(vocab_path):
    if os.path.isfile(vocab_path):
        tokenizer = torch.load(vocab_path)
        return tokenizer
    else:
        raise Exception("Invalid vocab path")