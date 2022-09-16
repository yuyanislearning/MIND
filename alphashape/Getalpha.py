import os
import sys
import pandas as pd
import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))
import alphashape
from Bio.PDB import *
from tqdm import tqdm
import pdb
from absl import app, flags
import time
from multiprocessing import Pool
from os.path import exists
from numba import njit

directory = '/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_updated'
out_dir = '/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_updated_alpha_shape'
parser = MMCIFParser()
thres = 0.3

def main(argv):
    
    # parser = PDBParser()
    a = os.popen('ls '+ directory)
    filenames = [(i, line.strip()) for i, line in enumerate(a)]
    # get_alpha_shape((0, 'AF-Q5S007-F1-model_v3.cif' ))

    with Pool(96) as p:
        print(p.map(get_alpha_shape, filenames))
        
def get_alpha_shape(filename):
    i, filename = filename
    uid = filename.split('-')[1]
    if exists(os.path.join(out_dir, uid+ '_alpha_shape_'+str(thres)+'.npy')):
        return None
    structure = parser.get_structure(uid, os.path.join(directory, filename))
    if len(structure[0])>1:
        print(uid, len(structure[0]))

    res = structure[0]['A'].get_residues()
    res_ls = [r for r in res if is_aa(r)]
    
    all_coord = []
    rm_ls = []
    atom2aa = {}# map atom index to aa index
    count=0
    for j,res in enumerate(res_ls):
        for atom in res.get_list():
            all_coord.append(atom.coord)
            if atom.name in ['N','C','CA','O']:
                rm_ls.append(count)
            atom2aa[count] = j
            count+=1
    
    res_dict = {tuple(r):i for i,r in enumerate(all_coord)}

    alpha_shape = alphashape.alphashape(all_coord, thres)
    graph = np.zeros((len(res_ls),len(res_ls)))
    # plot_shape(alpha_shape)
    # pdb.set_trace()
    for face in alpha_shape.faces:
        triple = [res_dict[tuple(alpha_shape.vertices[fa])] for fa in face]
        triple = [atom2aa[tr] for tr in triple if tr not in rm_ls]
        graph = add_triple(graph, triple)
    graph = graph.astype(int)
    np.fill_diagonal(graph, 1)
    fw = open(os.path.join(out_dir, uid+ '_alpha_shape_'+str(thres)+'.npy'),'bw')
    np.save(fw, graph)    



def find_res(chain):
    res_ls = []
    for r in chain:
        if is_aa(r):
            res_ls.append(r) 
    return res_ls

def add_triple(graph, triple):
    # assert len(triple)==3
    if len(triple)==0 or len(triple)==1:
        return graph
    if len(triple)==2 or len(triple)==3:

        graph[triple[0],triple[1]] = 1
        graph[triple[1],triple[0]] = 1
    if len(triple)==3:
        graph[triple[0],triple[2]] = 1
        graph[triple[2],triple[0]] = 1

        graph[triple[1],triple[2]] = 1
        graph[triple[2],triple[1]] = 1
    return graph

def plot_shape(alpha_shape):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)
    plt.show()
    plt.savefig('temp.png')

if __name__ == '__main__':
    app.run(main)
