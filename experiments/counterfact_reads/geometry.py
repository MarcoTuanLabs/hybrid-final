"""Fit a rank-one boundary correction to the four future read responses of a selected head pair."""
import numpy as np

def normalize(reads):
    return reads / np.sqrt(np.mean(reads**2,axis=1,keepdims=True) + 1e-6)

def choose_pair(scores):
    return 2 * int(scores.reshape(-1, 2).sum(axis=1).argmax())

def designs(delta, boundary_queries, background):
    responses = delta @ boundary_queries
    raw = np.sum(responses**2,axis=(1,2))
    contrast = normalize(background + responses) - normalize(background)
    selected = {
        "normalized": choose_pair(np.sum(contrast**2,axis=(1,2))),
        "raw": choose_pair(raw),
        "matrix": choose_pair(np.sum(delta**2,axis=(1,2))),
        "fixed": 28
    }
    selected["pair_dense"] = selected["normalized"]
    for method, start in selected.items():
        ds = delta[start:start+2]
        if method == "pair_dense":
            yield method,start,ds.astype(np.float32),ds.size*4
            continue
        w = boundary_queries[start:start+2]
        if method == "matrix":
            u,s,vh = np.linalg.svd(ds,full_matrices=False)
            right = vh[:,:1,:]
        else:
            u,s,vh = np.linalg.svd(ds@w,full_matrices=False)
            right = vh[:,:1,:] @ np.linalg.pinv(w,rcond=1e-7)
        left = (u[:,:,:1]*s[:,None,:1]).astype(np.float32)
        right = right.astype(np.float32)
        yield method,start,left@right,left.nbytes+right.nbytes
