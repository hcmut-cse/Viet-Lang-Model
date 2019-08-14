import torch

def seq2text(seq, tokenizer, end_token=0):
    if isinstance(seq, torch.Tensor):
        seq = seq.numpy()

    text = ''
    for s in seq:
        if s==end_token:
            break
        text += tokenizer.index_word[s]
    return text

