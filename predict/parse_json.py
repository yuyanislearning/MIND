import os
import json
import pdb
from Bio import SeqIO

fasta_file = '/local2/yuyan/PTM-Motif/PTM-pattern-finder/predict/JC/JC.fasta'
seq_dat = {}

f = open('/local2/yuyan/PTM-Motif/PTM-pattern-finder/predict/JC/res_OPTM.tsv','w')
with open('/local2/yuyan/PTM-Motif/Data/Musite_data/PTM_test/eval/OPTM_fifthteen_JC.json') as fr:
    dat = json.load(fr)

with open(fasta_file, 'r') as fp:
    seqs = list(SeqIO.parse(fp, 'fasta')) 
for rec in seqs:
    uid = rec.id.split('|')[1]
    sequence=str(rec.seq)
    seq_dat[uid] = sequence

f.write('\t'.join(['Uniprot_id','site','amino acid modified', 'PTM type','prediction score'])+'\n')
for k in dat:
    if float(dat[k])>0.5:
        prob = dat[k]
        k = k.split('_')
        site = int(k[1])
        aa = seq_dat[k[0]][site]
        if k[2]=='Lys-OH' or k[2]=='ProCH':
            continue
        f.write('\t'.join([k[0],str(int(k[1])+1),aa,k[2],prob])+'\n')
f.close()
