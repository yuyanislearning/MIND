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
import Bio
from numba import njit
import math

directory = '/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_updated'
out_dir = '/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_NF'
parser = MMCIFParser()

def main(argv):
    
    # parser = PDBParser()
    a = os.popen('ls '+ directory)
    filenames = [(i, line.strip()) for i, line in enumerate(a)]

    with Pool(96) as p:
        print(p.map(get_node_features, filenames))
        
def get_node_features(filename):
    i, filename = filename
    uid = filename.split('-')[1]
    if exists(os.path.join(out_dir, uid+ '_node_features.npy')):
        return None
    structure = parser.get_structure(uid, os.path.join(directory, filename))
    if len(structure[0])>1:
        print(uid, len(structure[0]))

    res_ls = structure[0]['A'].get_residues()
    res_ls = [r for r in res_ls if is_aa(r)]

    dihedral_features = get_dihedral(res_ls)
    dir_features = get_fr(res_ls)
    im_features = get_im(res_ls)
    final_features = np.concatenate([dihedral_features, dir_features, im_features], axis=1)
    
    fw = open(os.path.join(out_dir, uid+ '_node_features.npy'),'bw')
    np.save(fw, final_features)    

def get_fr(res_ls):
    rf_fea = np.zeros((len(res_ls),6))
    for i in range(len(res_ls)):
        if i==0:
            f = res_ls[i+1]['CA'].get_vector()- res_ls[i]['CA'].get_vector()
            # r = res_ls[i-1]['CA'].get_vector()- res_ls[i]['CA'].get_vector()
            f = f/np.linalg.norm(f)
            # r = r/np.linalg.norm(r)
            rf_fea[i,:] = np.concatenate([f.get_array(),np.zeros((3))],axis=0)
        elif i==len(res_ls)-1:
            # f = res_ls[i+1]['CA'].get_vector()- res_ls[i]['CA'].get_vector()
            r = res_ls[i-1]['CA'].get_vector()- res_ls[i]['CA'].get_vector()
            # f = f/np.linalg.norm(f)
            r = r/np.linalg.norm(r)
            rf_fea[i,:] = np.concatenate([np.zeros((3)),r.get_array()],axis=0)
        else:
            f = res_ls[i+1]['CA'].get_vector()- res_ls[i]['CA'].get_vector()
            r = res_ls[i-1]['CA'].get_vector()- res_ls[i]['CA'].get_vector()
            f = f/np.linalg.norm(f)
            r = r/np.linalg.norm(r)
            rf_fea[i,:] = np.concatenate([f.get_array(),r.get_array()],axis=0)
    return rf_fea

def get_im(res_ls):
    im_fea = np.zeros((len(res_ls),3))
    for i in range(len(res_ls)):
        n = res_ls[i]['N'].get_vector() - res_ls[i]['CA'].get_vector()
        c = res_ls[i]['C'].get_vector() - res_ls[i]['CA'].get_vector()
        nc = np.multiply(n.get_array(),c.get_array())
        npc = (n+c).get_array()
        im = math.sqrt(1/3)*(nc) / np.linalg.norm(nc) - math.sqrt(2/3)*(npc)/np.linalg.norm(npc)
        im = im/np.linalg.norm(im)
        im_fea[i,:]=im
    return im_fea

def get_dihedral(res_ls):
    di_fea = np.zeros((len(res_ls), 6))
    for i in range(len(res_ls)):
        if i==0:
            psi = Bio.PDB.vectors.calc_dihedral(res_ls[i]['N'].get_vector(),res_ls[i]['CA'].get_vector(),\
                res_ls[i]['C'].get_vector(), res_ls[i+1]['N'].get_vector() )
            omega = Bio.PDB.vectors.calc_dihedral(res_ls[i]['CA'].get_vector(),res_ls[i]['C'].get_vector(),\
                res_ls[i+1]['N'].get_vector(), res_ls[i+1]['CA'].get_vector() )
            di_fea[i,:] = np.array([0,0, math.sin(psi), math.cos(psi),\
                math.sin(omega), math.cos(omega)])                
        elif i==len(res_ls)-1:
            phi = Bio.PDB.vectors.calc_dihedral(res_ls[i-1]['C'].get_vector(),res_ls[i]['N'].get_vector(),\
                res_ls[i]['CA'].get_vector(), res_ls[i]['C'].get_vector() )
            di_fea[i,:] = np.array([math.sin(phi), math.cos(phi), 0,0,0,0])    
        else:
            phi = Bio.PDB.vectors.calc_dihedral(res_ls[i-1]['C'].get_vector(),res_ls[i]['N'].get_vector(),\
                res_ls[i]['CA'].get_vector(), res_ls[i]['C'].get_vector() )
            psi = Bio.PDB.vectors.calc_dihedral(res_ls[i]['N'].get_vector(),res_ls[i]['CA'].get_vector(),\
                res_ls[i]['C'].get_vector(), res_ls[i+1]['N'].get_vector() )
            omega = Bio.PDB.vectors.calc_dihedral(res_ls[i]['CA'].get_vector(),res_ls[i]['C'].get_vector(),\
                res_ls[i+1]['N'].get_vector(), res_ls[i+1]['CA'].get_vector() )
            di_fea[i,:] = np.array([math.sin(phi), math.cos(phi), math.sin(psi), math.cos(psi),\
                math.sin(omega), math.cos(omega)])
    return di_fea

def find_res(chain):
    res_ls = []
    for r in chain:
        if is_aa(r):
            res_ls.append(r) 
    return res_ls

def add_triple(graph, triple):
    assert len(triple)==3
    graph[triple[0],triple[1]] = 1
    graph[triple[1],triple[0]] = 1

    graph[triple[0],triple[2]] = 1
    graph[triple[2],triple[0]] = 1

    graph[triple[1],triple[2]] = 1
    graph[triple[2],triple[1]] = 1
    return graph

if __name__ == '__main__':
    app.run(main)
