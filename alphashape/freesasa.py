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
import freesasa
from Bio.PDB.SASA import ShrakeRupley

directory = '/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_updated'
out_dir = '/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_updated_freesasa'
parser = MMCIFParser()
thres = 0.3


def main(argv):
    
    # parser = PDBParser()
    a = os.popen('ls '+ directory)
    filenames = [(i, line.strip()) for i, line in enumerate(a)]
    # get_alpha_shape((0, 'AF-Q5S007-F1-model_v3.cif' ))
    with Pool(96) as p:
        print(p.map(get_freesasa, filenames))
        
def get_freesasa(filename):
    sr = ShrakeRupley()
    i, filename = filename
    uid = filename.split('-')[1]
    if exists(os.path.join(out_dir, uid+ '_freesasa.npy')):
        return None
    structure = parser.get_structure(uid, os.path.join(directory, filename))
    sr.compute(structure, level='R')
    res = structure[0]['A'].get_residues()
    res_ls = [r for r in res if is_aa(r)]
    sasa = np.array([r.sasa for r in res_ls])
    np.save(os.path.join(out_dir, uid+ '_freesasa.npy'),sasa)



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
