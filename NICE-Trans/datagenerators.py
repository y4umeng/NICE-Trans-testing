import os, sys
import numpy as np
import scipy.ndimage

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
            X = np.load(fixed, allow_pickle=True)
            X = np.reshape(X[:,:,:144], (144, 192, 160))
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
            X = np.load(moving, allow_pickle=True)
            X = np.reshape(X[:,:,:144], (144, 192, 160))
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


def load_volfile(datafile, np_var):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        if np_var == 'all':
            X = X = np.load(datafile)
        else:
            X = np.load(datafile)[np_var]

    return X


def print_gpu_usage(note=""):
    print(note)
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))