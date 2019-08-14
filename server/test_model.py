import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *
from beamsearch import *
from utils import *

fn_model_path = 'weight/fn_model.h5'
ln_model_path = 'weight/ln_model.h5'
fn_vocab_path = 'weight/fn_tokenizer.h5'
ln_vocab_path = 'weight/ln_tokenizer.h5'

fn_tok = torch.load(fn_vocab_path)
ln_tok = torch.load(ln_vocab_path)

fn_model = get_model(len(fn_tok.word_index)+1)
fn_model.load_state_dict(torch.load(fn_model_path))
ln_model = get_model(len(ln_tok.word_index)+1)
ln_model.load_state_dict(torch.load(ln_model_path))

inp = 'v'
seq = fn_tok.texts_to_sequences([inp])
seq = torch.tensor(seq)
fn_model.eval()
ln_model.eval()

end_token = 2

with torch.no_grad():
    seq, score = beam_search(seq, fn_model, -1, 5, 100)

sent = [seq2text(x, fn_tok, end_token) for x in seq]

print(sent)
print(score)
print(fn_tok.word_index['$'])