import numpy as np
import torch

def update_topk(topk_seq, topk_score, new_hs, new_cs, probs, k, ix):
    # topk_seq: kxS
    # topk_score: kx1
    # topk_hs, topk_cs: ?xkx?
    # probs: kx1xV
    topk_probs, topk_ix = probs[:,-1,:].topk(k)
    topk_log = topk_probs.log()
    new_scores = topk_log + topk_score
    
    k_probs, k_ix = new_scores.view(-1).topk(k)
    row = k_ix // k
    col = k_ix % k
    
    topk_seq[:, :ix] = topk_seq[row, :ix]
    topk_seq[:, ix] = topk_ix[row, col]
    
    topk_hs = new_hs[:,row,:]
    topk_cs = new_cs[:,row,:]
    
    topk_score = k_probs.unsqueeze(1)
    return topk_seq, topk_score, topk_hs, topk_cs


def beam_search(seq, model, end_token=0, beam_width=3, max_len=30):
    "Search word by current characters"
    # seq: tensor 1xS
    topk_seq = torch.zeros(beam_width, max_len).long()
    # topk_seq[:,:len(start)] = seq[0,:]
    prob, (hs, cs) = model.predict(seq) # 1xSxV, hs
    prob_k, idx_k = prob[:,-1,:].topk(k=beam_width, dim=-1)
    # cix = len(start)
    cix = 0
    topk_seq[:,cix] = idx_k[0]
    cix+=1
    topk_score = torch.zeros(beam_width, 1)
    topk_hs = torch.zeros(hs.size(0), beam_width, hs.size(2))
    topk_cs = torch.zeros(cs.size(0), beam_width, cs.size(2))
    topk_hs[:,:,:] = hs[:,-1,:].unsqueeze(1)
    topk_cs[:,:,:] = cs[:,-1,:].unsqueeze(1)
    
    res_seq = []
    res_score = []
    
    for i in range(cix, max_len):
        probs, (hs, cs) = model.predict(topk_seq[:,i-1].unsqueeze(-1), (topk_hs, topk_cs)) # 1xSxV, hs
        topk_seq, topk_score, topk_hs, topk_cs = update_topk(topk_seq, topk_score, hs, cs, probs, beam_width, i)

        eos_ix = (topk_seq==end_token).nonzero()
        
        seq_end_ix = np.unique(eos_ix[:,0].numpy())
        seq_notend_ix = [t for t in range(beam_width) if t not in seq_end_ix]
        
        if len(seq_end_ix) > 0:
            _seq = topk_seq[seq_end_ix]
            for s in _seq:
                res_seq.append(s.numpy())
            
            _score = topk_score[seq_end_ix]
            for s in _score:
                res_score.append(s.numpy()[0])
            
            topk_score = topk_score[seq_notend_ix].contiguous()
            topk_seq = topk_seq[seq_notend_ix].contiguous()
            topk_hs = topk_hs[:,seq_notend_ix,:].contiguous()
            topk_cs = topk_cs[:,seq_notend_ix,:].contiguous()
            
            beam_width -= len(seq_end_ix)
            
        if beam_width==0:
            break
    if len(res_seq) > 0:
        return res_seq, res_score
    return topk_seq.numpy(), topk_score.numpy()[:,0]