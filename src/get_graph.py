import json
from tqdm import tqdm
import numpy as np
from tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token, ALL_AAS
from multiprocessing import Pool
from os.path import exists
import pdb
from Bio import SeqIO

seq_len = 514
eval = True
chunk_size = 512
adj_dir = '/workspace/PTM/Data/Musite_data/Structure/pdb/AF_updated_cont_map/'

def get_dat():
    # with open('../../Data/Musite_data/ptm/PTM_test.json', 'r') as fp:
    #     # data structure: {PID:{seq, label:[{site, ptm_type}]}}
    #     dat = json.load(fp)
    #     print('loading data')
    records = []
    # for k in tqdm(dat):
    #     # some case that the data miss sequence, skip
    #     if dat[k].get('seq',-1)==-1:
    #         continue
    for line in open('/workspace/PTM/Data/PTMVar/cardiac_pro.txt'):
        line = line.strip()
        if line=='UID':
            continue
        records = cut_protein_no_label(line,records,  chunk_size=seq_len-2, eval=eval)

    # PTM label to corresponding amino acids 
    label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
    'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST',
    "Arg-OH_R":'R',"Asn-OH_N":'N',"Asp-OH_D":'D',"Cys4HNE_C":"C","CysSO2H_C":"C","CysSO3H_C":"C",
    "Lys-OH_K":"K","Lys2AAA_K":"K","MetO_M":"M","MetO2_M":"M","Phe-OH_F":"F",
    "ProCH_P":"P","Trp-OH_W":"W","Tyr-OH_Y":"Y","Val-OH_V":"V"}
    # get unique labels
    # unique_labels = sorted(set([l['ptm_type'] for d in records for l in d['label'] ]))
    # label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
    # index_to_label = {i: str(label) for i, label in enumerate(unique_labels)}

    seq = [d['seq'] for d in records]
    # label = [ d['label'] for d in records]
    uid = [d['uid'] for d in records]
    # pos_label = [d['pos_label'] for d in records]

    # save memory
    records = None
    

    # Tokenize the sequence
    X = tokenize_seqs(seq)


    with Pool(96) as p:
        p.map(get_graph, list(zip(X,uid)))

def cut_protein_no_label( uid,records,  chunk_size, eval):
    # cut the protein if it is longer than chunk_size
    # only includes labels within middle chunk_size//2
    # during training, if no pos label exists, ignore the chunk
    # during eval, retain all chunks for multilabel; retain all chunks of protein have specific PTM for binary
    assert chunk_size%4 == 0
    quar_chunk_size = chunk_size//4
    half_chunk_size = chunk_size//2
    with open('/workspace/PTM/Data/Musite_data/fasta/'+uid+'.fa', 'r') as fp:
        # data structure: {PID:{seq, label:[{site, ptm_type}]}}
        sequence = str(list(SeqIO.parse(fp, 'fasta'))[0].seq)
    
    if len(sequence) > chunk_size:
        label_count = 0
        for i in range((len(sequence)-1)//half_chunk_size):
            # the number of half chunks=(len(sequence)-1)//chunk_size+1,
            # minus one because the first chunks contains two halfchunks
            max_seq_ind = (i+2)*half_chunk_size
            if i==0:
                cover_range = (0,quar_chunk_size*3)
            elif i==((len(sequence)-1)//half_chunk_size-1):
                cover_range = (quar_chunk_size+i*half_chunk_size, len(sequence))
                max_seq_ind = len(sequence)
            else:
                cover_range = (quar_chunk_size+i*half_chunk_size, quar_chunk_size+(i+1)*half_chunk_size)

            records.append({
                'uid': uid+'~'+str(i),
                'seq': sequence[i*half_chunk_size: max_seq_ind],
            })                    
    else:
        records.append({
            'uid': uid,
            'seq': sequence,
            
        })            
    return records


def cut_protein( dat,records, k, chunk_size, eval):
    # cut the protein if it is longer than chunk_size
    # only includes labels within middle chunk_size//2
    # during training, if no pos label exists, ignore the chunk
    # during eval, retain all chunks for multilabel; retain all chunks of protein have specific PTM for binary
    assert chunk_size%4 == 0
    quar_chunk_size = chunk_size//4
    half_chunk_size = chunk_size//2
    sequence = str(dat[k]['seq'])
    labels = dat[k]['label']
    pos_label = list(set([lbl['ptm_type'] for lbl in labels]))
    if len(sequence) > chunk_size:
        label_count = 0
        for i in range((len(sequence)-1)//half_chunk_size):
            # the number of half chunks=(len(sequence)-1)//chunk_size+1,
            # minus one because the first chunks contains two halfchunks
            max_seq_ind = (i+2)*half_chunk_size
            if i==0:
                cover_range = (0,quar_chunk_size*3)
            elif i==((len(sequence)-1)//half_chunk_size-1):
                cover_range = (quar_chunk_size+i*half_chunk_size, len(sequence))
                max_seq_ind = len(sequence)
            else:
                cover_range = (quar_chunk_size+i*half_chunk_size, quar_chunk_size+(i+1)*half_chunk_size)
            sub_labels = [lbl for lbl in labels if (lbl['site'] >= cover_range[0] and lbl['site'] < cover_range[1])]
            if not eval:
                if len(sub_labels)==0:
                    continue
                records.append({
                    'uid': k+'~'+str(i),
                    'seq': sequence[i*half_chunk_size: max_seq_ind],
                    'label': sub_labels,
                    'pos_label': pos_label,
                })
            else:
                records.append({
                    'uid': k+'~'+str(i),
                    'seq': sequence[i*half_chunk_size: max_seq_ind],
                    'label': sub_labels,
                    'pos_label': pos_label,
                })                    
            label_count+=len(sub_labels)
    else:
        records.append({
            'uid': k,
            'seq': sequence,
            'label':labels,
            'pos_label': pos_label,
        })            
    return records


def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]

def get_graph( input ):
    X,uid = input
    adj_name = '../ttt/'+uid+'_'+str(514)+'_'+str(5)+'.npy'
    if not exists(adj_name):
        if '~' in uid:
            n_seq = int(uid.split('~')[1])
            tuid = uid.split('~')[0]
            
            if exists(adj_dir+tuid+'.cont_map.npy') :
                adj = np.load(adj_dir+tuid+'.cont_map.npy')
                n = adj.shape[0]
                left_slice = n_seq*512//2
                right_slice = min((n_seq+2)*512//2, n)
                adj = adj[left_slice:right_slice, left_slice:right_slice]
                    
                
            else:
                n = np.where(np.array(X)==24)[0][0]-1
                adj = np.zeros((n,n))
                adj = assign_neighbour(adj, 5)
        else:
            if exists(adj_dir+uid+'.cont_map.npy') :
                adj = np.load(adj_dir+uid+'.cont_map.npy')
            else:
                # 24 is the stop sign
                n = np.where(np.array(X)==24)[0][0]-1
                adj = np.zeros((n,n))
                adj = assign_neighbour(adj, 5)
        
        # adj = graph_encoding(adj, 20, 514)
        adj = pad_adj(adj, 514)
        np.save(adj_name,adj)

    return None


def graph_encoding(adj, encode_d, seq_len):
    D = np.diag(np.sum(adj,axis=1)** -0.5)
    L = np.eye(adj.shape[0]) - np.matmul(np.matmul(D ,adj),  D)
    # eigen vectors
    EigVal, EigVec = np.linalg.eig(L)

    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    if EigVec.shape[1]<encode_d:
        EigVec = np.concatenate((EigVec,np.zeros((EigVec.shape[0], encode_d - EigVec.shape[1]))), axis=1)
    else:
        EigVec = EigVec[:, 0:encode_d]
    if EigVec.shape[0] < seq_len:
        EigVec = np.concatenate((EigVec, np.zeros((seq_len - EigVec.shape[0], EigVec.shape[1]))), axis=0)
    return EigVec



def assign_neighbour(adj, num_cont):
    # assign the 2*num_cont neighbours to adj 
    n= adj.shape[0]
    if num_cont>=n:
        return np.ones((n,n), dtype=int)
    if num_cont==0:
        return adj
    for i in range(n):
        left = max(i-num_cont,0)
        right = min(i+num_cont+1, n)
        adj[i,left:right] = 1
    # adj = adj + np.tri(n,k=num_cont) - np.tri(n, k=-(num_cont+1))
    # adj[adj>0] = 1
    return adj

def pad_adj( adj, seq_len):
    pad_adj = np.zeros((seq_len, seq_len))
    pad_adj[1:(1+adj.shape[0]),1:(1+adj.shape[0])] = adj
    return pad_adj    

get_dat()