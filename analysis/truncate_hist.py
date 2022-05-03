import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

adj_dir = '/workspace/PTM/Data/Musite_data/Structure/pdb/AF_cont_map/'

def fill_adj_chunk(adj, chunk_size):
    # the number of half chunks=(len(sequence)-1)//chunk_size+1, minus one because the first chunks contains two halfchunks
    assert chunk_size%4 == 0
    half_chunk_size = chunk_size//2
    n = adj.shape[0]
    if n > chunk_size:
        for i in range((n-1)//half_chunk_size):
            left = i*half_chunk_size
            right = min((i+2)*half_chunk_size,n)
            adj[left:right, left:right] = 0
    return adj

def fill_adj_local(adj, fill_cont):
    # fill_cont *= 2
    n = adj.shape[0]
    if fill_cont>=n:
        return np.zeros((n,n), dtype=int)
    adj = adj - np.identity(n, dtype=int)
    for i in range(1,fill_cont):
        adj[i:n,0:(n-i)] = adj[i:n,0:(n-i)] - np.identity(n-i, dtype=int)
        adj[0:(n-i),i:n] = adj[0:(n-i),i:n] - np.identity(n-i, dtype=int)
    adj[adj<0] = 0
    return adj

local_his = []
global_his = []
all_his = []
fill_cont = 17
chunk_size = 512
for fle in tqdm(os.listdir(adj_dir)):
    if fle.endswith('npy'):
        adj = np.load(adj_dir+fle)
        all_his.append(np.sum(adj))
        # adj = fill_adj_local(adj, fill_cont)
        local_his.append(np.sum(adj))
        adj = fill_adj_chunk(adj, chunk_size)
        global_his.append(np.sum(adj))
        
local_his = np.array(local_his)
global_his = np.array(global_his)   
local_his = local_his-global_his
all_his = np.array(all_his)

local_total = np.sum(local_his)
global_total = np.sum(global_his)
all_total = np.sum(all_his)

with open('res/truncate_%d_%d.txt'%(fill_cont, chunk_size),'w') as fw:
    fw.write('total: %f\n'%(all_total))
    fw.write('local total: %f\n'%(local_total))
    fw.write('global total: %f\n'%(global_total))



