from PIL import Image
import numpy as np
import os
import os.path as osp
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


cmap = sio.loadmat('human_colormap.mat')['colormap'].T
newcmp = ListedColormap(cmap)

def npo(path, fn):
    return np.array(Image.open(osp.join(path, fn)))

def save_arr(arr, fn):
    (Image.fromarray(arr.astype(np.int8))).save(fn)

def sq(arr): return np.multiply(arr, arr)

def ps(arr): print(arr.shape)
def pu(arr): print(np.unique(arr))

def decode(arr, cmap):
    ones = np.ones((1,20))
    seg = np.abs((arr[:,:,:,np.newaxis].dot(ones))-cmap).sum(2)
    
    if seg.min(2).sum()!=0: return None
    return seg.argmin(2)

cmap_mask = np.zeros((256,192,3,20))
for i in range(256):
    for j in range(192):
        cmap_mask[i,j,:,:] = cmap

#ref_path = "data/%s/image-parse"
new_path = "seg/"
target_path = "seg_one/%s/"

for dirn in os.listdir(new_path):
    mode = "test" if "test" in dirn else "train"
    old_path = osp.join(new_path, dirn)
    tot_num = len(os.listdir(old_path))
    print()
    for i, fn in enumerate(os.listdir(old_path)):
        print("\r%d / %d"%(i, tot_num), end=" "*10)
        seg = decode(npo(old_path, fn)/256.0, cmap_mask)
        if seg is not None:
            save_arr(seg, (target_path%mode)+fn)
        else:
            print(old_path, fn)
