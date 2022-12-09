from absl import flags
from absl import logging
import tensorflow as tf
import pandas as pd
from sklearn.metrics import  confusion_matrix

import numpy as np
import random
import math
import json
from Bio import SeqIO
from os.path import exists
from tqdm import tqdm
import re
import os 

from .tokenization import additional_token_to_index,  tokenize_seq, parse_seq, aa_to_token_index, index_to_token, ALL_AAS


def handle_flags():
    flags.DEFINE_string("tflog",
            '3', "The setting for TF_CPP_MIN_LOG_LEVEL (default: 3)")
    flags.DEFINE_string('data_path',
            '../Data/Musite_data/ptm', 'path to data dir')
    flags.DEFINE_string('res_path',
            None, 'path to store output')            
    flags.DEFINE_string('adj_dir',
            '/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_updated_cont_map/', 'path to structure adjency matrix')
    flags.DEFINE_string('pretrain_name',
                'PTM', 'name of pretrain model')  
    flags.DEFINE_string('suffix', '', 'model name suffix')
    flags.DEFINE_string('class_weight_fle',
            '/local2/yuyan/PTM-Motif/PTM-pattern-finder/analysis/res/class_weights.json', 'path to class weights')
    flags.DEFINE_string('ptm_type', None, 'ptm type to predict')

    flags.DEFINE_bool("multilabel", True, "multilabel or not (default: True)")
    flags.DEFINE_bool("binary", False, "Binary or not (default: False)")
    flags.DEFINE_bool("class_weights", True, "use class weights or not (default: True)")
    flags.DEFINE_bool("graph", False, "use only partial sequence (default: False)")
    flags.DEFINE_bool("save_model", False, "save model (default: False)")
    flags.DEFINE_bool("pretrain", False, "use pretrain model (default: False)")
    flags.DEFINE_bool("ensemble", False, "ensemble learning")
    flags.DEFINE_bool("random_ensemble", False, "random batch ensemble learning")        
    flags.DEFINE_bool("inter", False, "for interpretation")

    # Training parameters.
    flags.DEFINE_integer("seq_len", 514, "maximum lenth+2 of the model sequence (default: 512)")
    flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 32)")# 128 for 514; 32 for 1026; 8 for 2050
    flags.DEFINE_integer("num_epochs",
            300, "Number of training epochs (default: 20)")
    flags.DEFINE_integer("n_lstm", 3, "number of lstm layer for rnn model")
    flags.DEFINE_integer("fill_cont", 5, "how many sequence should be considered as neighbour/2")
    flags.DEFINE_integer("n_fold", 5, "k-fold cross validation")
    flags.DEFINE_integer("d_model", 128, "k-fold cross validation")
    flags.DEFINE_integer("site", None, "site for prediction")


    flags.DEFINE_integer('random_seed',
            252, 'Random seeds for reproducibility (default: 252)')
    flags.DEFINE_float('learning_rate',
            1e-3, 'Learning rate while training (default: 1e-3)')
    FLAGS = flags.FLAGS


class PTMDataGenerator(tf.keras.utils.Sequence): #MetO_M   
    def __init__(self, file_name, FLAGS, shuffle=True,ind=None, eval=False, binary=False, class_weights=None, val=False):
        self.FLAGS = FLAGS
        records = []
        self.ind = ind
        self.eval = eval
        self.shuffle = shuffle
        self.chunk_size = FLAGS.seq_len-2
        self.graph = FLAGS.graph
        # self.num_cont = FLAGS.fill_cont
        self.d_model = FLAGS.d_model
        self.batch_size = FLAGS.batch_size    
        self.model = FLAGS.model
        self.binary = binary
        self.class_weights = class_weights
        self.seq_len = FLAGS.seq_len
        self.spar = True
        self.ensemble = FLAGS.ensemble
        self.random_ensemble = FLAGS.random_ensemble
        self.fill_cont = FLAGS.fill_cont
        if val:
            return None

        with open(file_name, 'r') as fp:
            # data structure: {PID:{seq, label:[{site, ptm_type}]}}
            dat = json.load(fp)
            print('loading data')
            for dat_count, k in tqdm(enumerate(dat)):
                # some case that the data miss sequence, skip
                if dat[k].get('seq',-1)==-1:
                    continue
                if k=='Q95JN5':
                    continue
                records = self.cut_protein(dat,records, k, chunk_size=FLAGS.seq_len-2, eval=eval)
        if self.ensemble  and not eval:
            file_name = FLAGS.data_path+'/PTM_val.json'
            with open(file_name, 'r') as fp:
                dat = json.load(fp)
                print('loading data')
                for k in tqdm(dat):
                    # some case that the data miss sequence, skip
                    if dat[k].get('seq',-1)==-1:
                        continue
                    records = self.cut_protein(dat,records, k, chunk_size=FLAGS.seq_len-2, eval=eval)
        
        # PTM label to corresponding amino acids 
        self.label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
        'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST',
        "Arg-OH_R":'R',"Asn-OH_N":'N',"Asp-OH_D":'D',"Cys4HNE_C":"C","CysSO2H_C":"C","CysSO3H_C":"C",
        "Lys-OH_K":"K","Lys2AAA_K":"K","MetO_M":"M","MetO2_M":"M","Phe-OH_F":"F",
        "ProCH_P":"P","Trp-OH_W":"W","Tyr-OH_Y":"Y","Val-OH_V":"V"}
        # get unique labels
        self.unique_labels = sorted(set([l['ptm_type'] for d in records for l in d['label'] ]))

        
        self.label_to_index = {str(label): i for i, label in enumerate(self.unique_labels)}
        self.index_to_label = {i: str(label) for i, label in enumerate(self.unique_labels)}

        self.seq = [d['seq'] for d in records]
        self.label = [ d['label'] for d in records]
        self.uid = [d['uid'] for d in records]
        self.pos_label = [d['pos_label'] for d in records]

        # save memory
        records = None
        

        # Tokenize the sequence
        self.X = tokenize_seqs(self.seq)

        # construct the raw Y and sample weights
        self.Y = []
        self.sample_weights = []
        print('Constructing Y and sample weights')
        y_keep = []
        for i in tqdm(range(len(self.X))):# create label matrix Y, and sample weight matrix
            # putting positive samples
            Y = np.zeros((len(self.X[i]), len(self.unique_labels))) if not self.binary \
                else np.zeros((len(self.X[i]), 1))
            sample_weights = np.zeros((len(self.X[i]), len(self.unique_labels))) if not self.binary \
                else np.zeros((len(self.X[i]), 1))
            for lbl in self.label[i]:
                if '~' in self.uid[i]:
                    n_seq = int(self.uid[i].split('~')[1])
                    site_num = int(lbl['site'])-n_seq*self.chunk_size//2+1 # minus i * half chunk size
                else:
                    site_num = int(lbl['site'])+1
                if self.binary:
                    if lbl['ptm_type'] == self.unique_labels[self.ind]:
                        Y[site_num, 0] = 1 
                        sample_weights[site_num, 0] = class_weights[lbl['ptm_type']][0] \
                            if (not self.neg_sam) and (not self.eval) else 1
                    else:
                        continue
                else:
                    Y[site_num, self.label_to_index[lbl['ptm_type']]] = 1 
                    sample_weights[site_num, self.label_to_index[lbl['ptm_type']]] = class_weights[lbl['ptm_type']][0] \
                            if (not self.neg_sam) and (not self.eval) else 1

            if self.binary:
                if self.unique_labels[self.ind] not in self.pos_label[i]:
                    continue
                if np.sum(Y)==0 and not self.eval:
                    continue
                else:
                    y_keep.append(i) # save index for X and uid
                    # pos_y = np.sum(Y, axis = 1).astype(int) >0
                    # Y = Y[pos_y]
                    # sample_weights = sample_weights[pos_y]
                    self.Y.append(Y)
                    self.sample_weights.append(sample_weights)
            else:
                self.Y.append(Y)
                self.sample_weights.append(sample_weights)                
        if self.binary:
            self.X = [self.X[i] for i in y_keep]
            self.uid = [self.uid[i] for i in y_keep]
            self.seq = [self.seq[i] for i in y_keep]
            self.label = [self.label[i] for i in y_keep]
        self.list_id = np.arange(len(self.uid))
        self.dat_len = len(self.uid)
        # get graph
        if self.graph:
            self.get_graph()


        logging.info('Loaded {} records from {}.'.format(len(self.seq),file_name))


        self.on_epoch_end()

    def get_graph(self):
        print('constructing graphs')
        for i in tqdm(range(len(self.uid))):
            adj_name = './temp/'+self.uid[i]+'_'+str(self.FLAGS.seq_len)+'_'+str(self.FLAGS.fill_cont)+'.npy'
            if not exists(adj_name):
                if '~' in self.uid[i]:
                    n_seq = int(self.uid[i].split('~')[1])
                    tuid = self.uid[i].split('~')[0]

                    if exists(self.FLAGS.adj_dir+tuid+'.cont_map.npy'):
                        adj = np.load(self.FLAGS.adj_dir+tuid+'.cont_map.npy')
                        n = adj.shape[0]
                        left_slice = n_seq*self.chunk_size//2
                        right_slice = min((n_seq+2)*self.chunk_size//2, n)
                        adj = adj[left_slice:right_slice, left_slice:right_slice]
                    else:
                        n = np.where(np.array(self.X[i])==24)[0][0]-1
                        adj = np.zeros((n,n))
                        adj = assign_neighbour(adj, self.FLAGS.fill_cont)
                else:
                    if exists(self.FLAGS.adj_dir+self.uid[i]+'.cont_map.npy'):
                        adj = np.load(self.FLAGS.adj_dir+self.uid[i]+'.cont_map.npy')
                        adj = rm_diag(adj,self.FLAGS.fill_cont)
                    else:
                        # 24 is the stop sign
                        n = np.where(np.array(self.X[i])==24)[0][0]-1
                        adj = np.zeros((n,n))
                        adj = assign_neighbour(adj, self.FLAGS.fill_cont)
                adj = pad_adj(adj, self.FLAGS.seq_len)
                
                np.save(adj_name,adj)
            else:
                next
                # try:
                #     adj = np.load(adj_name)
                # except:
                #     os.system('rm '+ adj_name)

        return None

    def train_val_split(self):
        # splitting training and validation
        
        fold = self.FLAGS.n_fold
        split_size = self.dat_len // 5 if self.random_ensemble else self.dat_len // fold # for random, every time get 1/5
        all_id = set(self.list_id)
        folds = []
        count = 0
        fail = True
        while fail:
            for _ in range(fold-1):
                temp_split = random.sample(all_id, split_size)
                folds.append(temp_split)
                if not self.random_ensemble:
                    # if not random, remove the set that already select
                    all_id = all_id - set(temp_split)
            if self.random_ensemble:
                temp_split = random.sample(all_id, split_size)
                folds.append(temp_split)
            else:
                folds.append(list(all_id))
            fail=False
            for fo in folds:
                Y = [self.Y[s] for s in fo]
                Y = np.sum(np.stack([np.sum(y, axis = 0) for y in Y]),axis=0)
                if any([y==0 for y in Y]):
                    fail=True
                    break
                
            count+=1
        self.folds = folds

    def init_fold(self, iter):
        # choose the corresponding list id and split out validation dataset
        self.list_id = np.array(list(set(np.arange(len(self.uid))) - set(self.folds[iter])))
        return self.split_val(np.array(self.folds[iter]))
    
    def split_val(self, val_idx):
        val_aug = PTMDataGenerator(None, self.FLAGS, shuffle=True, eval=True, binary=self.binary, val=True)
        val_aug.label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
        'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST',
        "Arg-OH_R":'R',"Asn-OH_N":'N',"Asp-OH_D":'D',"Cys4HNE_C":"C","CysSO2H_C":"C","CysSO3H_C":"C",
        "Lys-OH_K":"K","Lys2AAA_K":"K","MetO_M":"M","MetO2_M":"M","Phe-OH_F":"F",
        "ProCH_P":"P","Trp-OH_W":"W","Tyr-OH_Y":"Y","Val-OH_V":"V"}
        # get unique labels
        val_aug.unique_labels = self.unique_labels
        val_aug.label_to_index = self.label_to_index
        val_aug.index_to_label = self.index_to_label
        val_aug.seq = [self.seq[idx] for idx in val_idx]
        val_aug.label = [self.label[idx] for idx in val_idx]
        val_aug.uid = [self.uid[idx] for idx in val_idx]
        val_aug.pos_label = [self.pos_label[idx] for idx in val_idx]
        val_aug.list_id = np.arange(len(val_aug.uid))
        # Tokenize the sequence
        val_aug.X = [self.X[idx] for idx in val_idx]
        # construct the raw Y and sample weights
        val_aug.Y = [self.Y[idx] for idx in val_idx]
        val_aug.sample_weights = [self.sample_weights[idx] for idx in val_idx]
        val_aug.on_epoch_end()
        return val_aug

    def update_unique_labels(self, unique_labels):
        self.unique_labels = unique_labels
    
    def __len__(self):
        # number of batch per epoch
        return math.ceil(len(self.index) / self.batch_size )

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.list_id[k] for k in index]
        X, y, sample_weights = self.__get_data(batch)
        return (X, y, sample_weights)

    def on_epoch_end(self):
        self.index = np.arange(len(self.list_id))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        # Tokenize X
        # X = [x[batch] for x in self.X]  # get batch for all component of X
        X = [self.X[b] for b in batch]
        Y = [self.Y[b] for b in batch]
        seqs = [self.seq[b] for b in batch]
        sample_weights = [self.sample_weights[b] for b in batch]
        uid = [self.uid[b] for b in batch]
        pos_labels = [self.pos_label[b] for b in batch]
        # padding for X, Y, sw
        max_batch_seq_len = self.seq_len
        X = self.pad_X(X, max_batch_seq_len)
        Y = self.pad_Y_sw(Y, max_batch_seq_len)
        sample_weights = self.pad_Y_sw(sample_weights, max_batch_seq_len)

        
        for i in range(len(X)):
            seq = [index_to_token[j] for j in X[i]]
            if self.eval:
                # for evaluation, get all negative sample
                if self.binary:
                    u = self.unique_labels[self.ind]
                    aa = self.label2aa[u]
                    if '~' in uid[i]:
                        n_seq = int(uid[i].split('~')[1])
                        neg_index = self.get_neg_cut(n_seq, seq, self.chunk_size, aa)
                    else:
                        neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa])
                    if len(neg_index)==0:
                        continue
                    if (not self.neg_sam) and (not self.eval):
                        sample_weights[i, neg_index, 0] += self.class_weights[u][1]
                    else:
                        sample_weights[i, neg_index, 0] = 1    

                else:
                    # multi labels
                    for u in self.unique_labels:
                        if u not in pos_labels[i]: # if the sequence doesn't has such pos label
                            continue
                        aa = self.label2aa[u]
                        if '~' in uid[i]:
                            n_seq = int(uid[i].split('~')[1])
                            neg_index = self.get_neg_cut(n_seq, seq, self.chunk_size, aa)
                        else:
                            neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa])
                        if len(neg_index)==0:
                            continue
                        if (not self.neg_sam) and (not self.eval):
                            sample_weights[i, neg_index, self.label_to_index[u]] += self.class_weights[u][1]
                        else:
                            sample_weights[i, neg_index, self.label_to_index[u]] = 1   
            else:
                # for training, get randomly selected neg samples.
                if self.binary:
                    # only get the specific PTM negative 
                    u = self.unique_labels[self.ind]
                    aa = self.label2aa[u]
                    if '~' in uid[i]:
                        n_seq = int(uid[i].split('~')[1])
                        all_neg_index = self.get_neg_cut(n_seq, seq, self.chunk_size, aa)
                    else:
                        all_neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa], dtype=int)
                
                    # remove positive labels
                    pos_index = set(np.array([j for j in range(Y.shape[1]) if Y[i,j, 0]==1], dtype = int))
                    if len(pos_index)==0:
                        continue
                    all_neg_index = set(all_neg_index) - pos_index
                    all_neg_index = np.array([j for j in all_neg_index])
                    # sampling
                    n_pos = int(np.sum(sample_weights[i,:,0]))
                    if len(all_neg_index)==0 or n_pos==0:
                        continue
                    if n_pos <= len(all_neg_index):#if more neg than neg
                        neg_index = np.array(random.sample(set(all_neg_index), k=n_pos), dtype=int)#n_pos TODO
                    else:
                        neg_index = all_neg_index#[0]
                    sample_weights[i, neg_index, 0] = 1
                else:
                    all_neg_indexs = []
                    for u in self.unique_labels:
                        aa = self.label2aa[u]
                        if '~' in uid[i]:
                            n_seq = int(uid[i].split('~')[1])
                            all_neg_index = self.get_neg_cut(n_seq, seq, self.chunk_size, aa)
                        else:
                            all_neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa], dtype=int)
                    
                        # remove positive labels
                        pos_index = set(np.array([j for j in range(Y.shape[1]) if Y[i,j, self.label_to_index[u]]==1], dtype = int))
                        all_neg_index = set(all_neg_index) - pos_index
                        all_neg_index = np.array([j for j in all_neg_index])
                        # sampling
                        n_pos = int(np.sum(sample_weights[i,:,self.label_to_index[u]]))
                        if len(all_neg_index)<1 or n_pos==0:
                            continue
                        if n_pos <= len(all_neg_index):#if more pos than neg
                            neg_index = np.array(random.sample(set(all_neg_index), k=n_pos), dtype=int)#n_pos TODO
                        else:
                            neg_index = all_neg_index#[0]
                        sample_weights[i, neg_index, self.label_to_index[u]] = 1
                    # add augmented data to sample weights and X
                   
        y = Y.reshape((Y.shape[0],-1,1))
        sample_weights = sample_weights.reshape((sample_weights.shape[0], -1,1))
        X = [X]
        
        if self.graph:    
            adjs = np.array([np.load('./temp/'+id+'_'+str(self.seq_len)+'_'+str(self.fill_cont)+'.npy', allow_pickle=True) for id in uid])
            X.append(adjs)
        

        return (X, y, sample_weights)

    def cut_protein(self, dat,records, k, chunk_size, eval):
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
    
    def get_neg_cut(self, n_seq, seq, chunk_size, aa):
        # get negative sample in cut proteins
        # n_seq the id of cut pieces in the protein
        if n_seq==0:
            # first chunk includes the first quarter chunk
            neg_index = np.array([j for j,amino in enumerate(seq[0:chunk_size//4*3]) if amino in aa])
        elif '<PAD>' in seq:
            # last chunk includes the last leftover chunk
            neg_index = np.array([j for j,amino in enumerate(seq[chunk_size//4:len(seq)]) if amino in aa])+ chunk_size//4
        else:
            # other chunk takes only the middle half chunk to avoid duplication in negative sample
            neg_index = np.array([j for j,amino in enumerate(seq[chunk_size//4: chunk_size//4*3]) if amino in aa])+ chunk_size//4        
        return neg_index
    
    def pad_X(self, X, seq_len):
        return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in X])
    
    def pad_Y_sw(self, Y, seq_len):
        num_label = len(self.unique_labels) if not self.binary else 1
        return np.array([np.concatenate((y, np.zeros((seq_len - len(y), num_label))))  for y in Y])
    

def pad_adj( adj, seq_len):
    pad_adj = np.zeros((seq_len, seq_len))
    pad_adj[1:(1+adj.shape[0]),1:(1+adj.shape[0])] = adj
    return pad_adj

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

def rm_diag(adj,fill_cont):
    n = adj.shape[0]
    if fill_cont>=n:
        return np.zeros((n,n), dtype=int)
    else:
        for i in range(n):
            left = max(i-fill_cont,0)
            right = min(i+fill_cont+1, n)
            adj[i,left:right] = 1
    return adj

def get_unique_labels(train_set, valid_set, test_set):
    return sorted( set([l['ptm_type'] for d in train_set.records for l in d['label'] ]).union(\
                set([l['ptm_type'] for d in valid_set.records for l in d['label'] ])).union(\
                    set([l['ptm_type'] for d in test_set.records for l in d['label'] ])))


def get_class_weights(train_set, valid_set, test_set, unique_labels):
    class_weights = {u:0 for u in unique_labels}

    for d in train_set.records:
        for l in d['label']:
            class_weights[l['ptm_type']]+=1 
    
    for d in valid_set.records:
        for l in d['label']:
            class_weights[l['ptm_type']]+=1 
    
    for d in test_set.records:
        for l in d['label']:
            class_weights[l['ptm_type']]+=1 
    
    class_weights = [float(class_weights[u]) for u in unique_labels]
    mean_w = sum(class_weights)/len(class_weights)
    class_weights = np.array([mean_w/u for u in class_weights])
    return class_weights

def adj2edge(adj):
    x,y = np.where(adj==1)
    return np.stack(x,y)


def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]

def retri_seq(site, half_len, seq, other_token_index):
    if site < half_len:
        left_s = seq[0:site+1]
    else:
        left_s = seq[(site-half_len):(site+1)]
    if site + half_len > len(seq):
        right_s = seq[(site+1):len(seq)]
    else:
        right_s = seq[(site+1):(site+half_len+1)]
    left_s = [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(left_s)]
    right_s = [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(right_s)] + [additional_token_to_index['<END>']]
    left_s = (half_len+2 - len(left_s)) * [additional_token_to_index['<PAD>']] + left_s
    right_s = right_s + (half_len+1 - len(right_s)) * [additional_token_to_index['<PAD>']]
    s = left_s + right_s    

    return s

def cut_protein(sequence, seq_len, aa):
    # cut the protein if it is longer than chunk_size
    # only includes labels within middle chunk_size//2
    # during training, if no pos label exists, ignore the chunk
    # during eval, retain all chunks for multilabel; retain all chunks of protein have specific PTM for binary
    chunk_size = seq_len - 2
    assert chunk_size%4 == 0
    quar_chunk_size = chunk_size//4
    half_chunk_size = chunk_size//2
    records = []
    if len(sequence) > chunk_size:
        for i in range((len(sequence)-1)//half_chunk_size):
            # the number of half chunks=(len(sequence)-1)//chunk_size+1,
            # minus one because the first chunks contains two halfchunks
            max_seq_ind = (i+2)*half_chunk_size
            if i==0:
                cover_range = (0,quar_chunk_size*3)
            elif i==((len(sequence)-1)//half_chunk_size-1):
                cover_range = (quar_chunk_size, len(sequence)-i*half_chunk_size)
                max_seq_ind = len(sequence)
            else:
                cover_range = (quar_chunk_size, quar_chunk_size+half_chunk_size)
            seq = sequence[i*half_chunk_size: max_seq_ind]
            idx = [j for j in range(len((seq))) if (seq[j] in aa and j >= cover_range[0] and j < cover_range[1])]
            records.append({
                'chunk_id': i,
                'seq': seq,
                'idx': idx
            })
    else:
        records.append({
            'chunk_id': None,
            'seq': sequence,
            'idx': [j for j in range(len((sequence))) if sequence[j] in aa]
        })
    return records


def ensemble_get_weights(PR_AUCs, unique_labels):
    weights = {ptm:None for ptm in unique_labels}
    for ptm in unique_labels:
        weight = np.array([PR_AUCs[str(i)][ptm] for i in range(len(PR_AUCs))])
        weight = weight/np.sum(weight)
        weights[ptm] = weight
    return weights # {ptm_type}


class CategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes    

        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight):     

        y_mask = tf.reshape(sample_weight, [-1, 13])
        y_true = tf.reshape(y_true,[-1,  13])
        y_pred = tf.reshape(y_pred,[-1,  13])

        y_trues = [y_true[:,i][y_mask[:,i]==1] for i in range(13)]
        y_preds = [y_pred[:,i][y_mask[:,i]==1] for i in range(13)]

        for i in range(13):
            print(pd.DataFrame(confusion_matrix(y_trues[i], y_preds[i]>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1']))

    def result(self):

        return None#self.cat_true_positives
