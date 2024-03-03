import os 
from Bio.PDB import *
import numpy as np
from tqdm import tqdm
import pdb
from os.path import exists
from absl import app, flags
import time
from multiprocessing import Pool
import argparse

# create a parser object
argparser = argparse.ArgumentParser(description='PDB to contact map')
# set the parser
argparser.add_argument('--pdb_dir', type=str,  help='pdb directory')
argparser.add_argument('--out_dir', type=str,  help='output directory')
argparser.add_argument('--thres', type=int, default=10 , help='threshold for contact map')
argparser.add_argument('--nproc', type=int, default=96 , help='number of processors')
# parse the arguments
args = argparser.parse_args()

# parser = PDBParser()
parser = MMCIFParser()

def main():
    t0 = time.time()
    # parser = PDBParser()
    a = os.popen('ls '+ args.pdb_dir)
    filenames = [line.strip() for i, line in enumerate(a)]
    with Pool(args.nproc) as p:
        print(p.map(getadj, filenames))
    print(time.time()-t0)
        

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def find_res(chain):
    res_ls = []
    for r in chain:
        if is_aa(r):
            res_ls.append(r) 
    return res_ls


def getadj(filename):
    uid = filename.split('-')[1]
    # AF-A0A023FFB5-F1-model_v3.cif
    # filename = 'AF-'+line+'-F1-model_v1.pdb'
    # chain = line[2].split('/')[0]# use the first chain
    # start = line[3]
    # end = line[4]
    # filename = pdb_name + '.pdb'
    # #print(os.path.join(directory, filename))
    structure = parser.get_structure(uid, os.path.join(args.pdb_dir, filename))
    # # retain the longest chain
    # chains = [ch.get_residues() for ch in structure[0]]
    # chains = [[res for res in ch if is_aa(res)] for ch in chains]
    # chain_l = [len(ch) for ch in chains]
    # chain_id = chain_l.index(max(chain_l))
    # chain = chains[chain_id]
    if len(structure[0])>1:
        print(uid, len(structure[0]))
        return None
    res = structure[0]['A'].get_residues()
    res = [r for r in res if is_aa(r)]

    res_ls = find_res(res)
    dist_map = calc_dist_matrix(res_ls, res_ls)
    cont_map = dist_map < args.thres
    cont_map = cont_map.astype(int)
    fw = open(os.path.join(args.out_dir, uid+ '.cont_map.npy'),'bw')
    np.save(fw, cont_map)
    



if __name__ == '__main__':
    app.run(main)