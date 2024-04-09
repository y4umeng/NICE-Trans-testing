import os, sys
import numpy as np
import scipy.ndimage
import torch
import pickle

def pkload(fname):
    with open(fname, 'rb') as f:
        f =  pickle.load(f)
        print(f"pickle: {type(f)}", flush=True)
        return f

def gen_s2s(gen, batch_size=1):

    while True:
        X = next(gen)
        fixed = X[0]
        moving = X[1]
        
        # generate a zero tensor as pseudo labels
        Zero = np.zeros((1))
        
        yield ([fixed, moving], [fixed, Zero, fixed, Zero])
        

def gen_pairs(pairs, batch_size=1):
    
    pairs_num = len(pairs)  
    while True:
        idx1 = np.random.randint(pairs_num, size=batch_size)
        idx2 = np.random.randint(pairs_num, size=batch_size)

        # load fixed images
        X_data = []
        for idx in idx1:
            fixed = pairs[idx]
            X = load_volfile(fixed)
            X = X[:,:,:144]
            X = np.reshape(X, (144, 192, 160))
            X = X[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # load moving images
        X_data = []
        for idx in idx2:
            moving = pairs[idx]
            X = load_volfile(moving) 
            X = X[:,:,:144]
            X = np.reshape(X, (144, 192, 160))
            X = X[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
        
        yield tuple(return_vals)

        
def load_by_name(path, name):
    
    npz_data = load_volfile(path+bytes.decode(name), np_var='all')
    
    X = npz_data['vol']
    X = X[np.newaxis, np.newaxis, ...]
    return_vals = [X]
    
    X = npz_data['label']
    X = X[np.newaxis, np.newaxis, ...]
    return_vals.append(X)
    
    return tuple(return_vals)


def load_volfile(datafile):
    with open(datafile, 'rb') as f:
        f =  pickle.load(f)
        print(f"pickle: {type(f)}", flush=True)
        print(f"pickle[0]: {type(f[0])}", flush=True)
        return f
    # np.load(moving, allow_pickle=True)[0]


def print_gpu_usage(note=""):
    print(note)
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))