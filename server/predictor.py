import torch
import numpy as np

from model import get_model, load_model, load_vocab
from beamsearch import beam_search
from utils import seq2text

class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Predictor(metaclass=Singleton):
    def init_model(self, fn_model_path, ln_model_path, fn_vocab_path, ln_vocab_path):
        self.fn_tokenizer = load_vocab(fn_vocab_path)
        self.ln_tokenizer = load_vocab(ln_vocab_path)
        self.fn_model = get_model(len(self.fn_tokenizer.word_index) + 1)
        self.ln_model = get_model(len(self.ln_tokenizer.word_index) + 1)
        load_model(self.fn_model, fn_model_path)
        load_model(self.ln_model, ln_model_path)
        self.fn_model.eval()
        self.ln_model.eval()


    def predict_last_name(self, inp, k=10, max_len=40, len_norm=False):
        seq = self.ln_tokenizer.texts_to_sequences([inp])
        seq = torch.tensor(seq).long()
        end_token = self.ln_tokenizer.word_index['$']
        with torch.no_grad():
            kseq, kscore = beam_search(seq, self.ln_model, end_token, k, max_len)
        ksent = [seq2text(x, self.ln_tokenizer, end_token) for x in kseq]

        kscore = np.exp(kscore)

        if len_norm:
            sent_len = [len(x) for x in ksent]
            kscore = kscore / sent_len
        
        res = [(sent,float(score)) for sent, score in zip(ksent, kscore)]
        return res
   
    def predict_first_name(self, inp, k=10, max_len=40, len_norm=False):
        seq = self.fn_tokenizer.texts_to_sequences([inp])
        seq = torch.tensor(seq).long()
        end_token = self.fn_tokenizer.word_index['$']
        with torch.no_grad():
            kseq, kscore = beam_search(seq, self.fn_model, end_token, k, max_len)
        ksent = [seq2text(x, self.ln_tokenizer, end_token) for x in kseq]

        kscore = np.exp(kscore)

        if len_norm:
            sent_len = [len(x) for x in ksent]
            kscore = kscore / sent_len
        
        res = [(sent,float(score)) for sent, score in zip(ksent, kscore)]
        return res

