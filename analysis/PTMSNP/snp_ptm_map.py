import json
from pprint import pprint
from tqdm import tqdm
import pdb
# SNPs = {'P51530':[('R','198','H')],'O75489':[('R','199','W')],'O00217':[('R','102','H')],
#     'Q9UMS0':[('G','190','R'),('G','189','R')],'Q8TB37':[('D','105','Y')],'Q9P2R7':[('R','284','C')],
#     'P15056':[('F','468','S'),('K','601','Q'),('T','599','R'),('K','601','Q')],
#     'P02545':[('R','60','G'),('R','435','C'),('I','210','S')],
#     'Q14896':[('P','608','L'),('Q','998','R'),('R','272','C')],'P12883':[('G','425','R')],'P10916':[('E','22','K')],
#     'Q86TC9':[('R','955','W'),('P','961','L')],'P04049':[('A','237','T')],'O75792':[('R','186','W')],
#     'P19429':[('R','162','P'),('S','166','F'),('R','162','Q')],'P09493':[('E','40','K')],
#     'Q9P2R6':[('P','1262','R')],'Q01484':[('R','3906','W')],'Q13936':[('E','477','K')],
#     'Q12809':[('P','251','S'),('S','320','L'),('D','323','N'),('R','1033','W')],'Q14524':[('R','569','W')]}

with open('/local2/yuyan/PTM-Motif/Data/PTMVar/all_PTMVar.json') as f:
    PTMVar = json.load(f)

ftr = open('/local2/yuyan/PTM-Motif/Data/PTMVar/cardiac_snps.txt')
count=0
snp_count = 0
fw = open('/local2/yuyan/PTM-Motif/Data/PTMVar/cardiac_snp_res.txt','w')
for line in tqdm(ftr):
    if count==0:
        count+=1
        continue
    line = line.strip().split('\t')
    uid = line[0]
    snp = (line[1],line[2],line[3])
    
    ptms = PTMVar[uid]
    for ptm in ptms:
        if ptm[0]==snp[0] and ptm[1] == snp[1] and ptm[2]==snp[2]:
            ptm_site = ptm[3]
            ptm_type = ptm[5]
            break

    OPTM=False
    if OPTM:
        suffix='_OPTM'
    else:
        suffix=''
# SNPs = {'O94759':[('V','934','I')], 'P02649':[('C','130','R')], 'Q5S007':[('G','2019','S')],
#     'Q8TF76':[('V','76','E')],'Q96RQ3':[('H','464','P')],'Q9BSA9':[('M','393','T')],'Q9C0K1':[('A','391','T')]}
# uid = 'Q14896'
# snps = [("Q","998","R"),("P","608","L"),("R","272","C")]
# for uid in SNPs:#SNPs:
    # for snp in SNPs[uid]:
    
    with open('/local2/yuyan/PTM-Motif/Data/PTMVar/cardiac/res/'+uid+'_'+snp[0]+snp[1]+snp[2]+suffix+'.json') as f:
        mut = json.load(f)

    with open('/local2/yuyan/PTM-Motif/Data/PTMVar/cardiac/res/'+uid+suffix+'.json') as f:
        wt = json.load(f)

    k = '_'.join([ptm_site, ptm_type])
    try:
        wt[k]
        mut[k]
    except:
        print(uid)
        print(k)
        continue
    if float(wt[k])>0.5 and float(mut[k])<0.5:
        snp_count+=1
        fw.write('\t'.join([uid, snp[0], snp[1], snp[2], ptm_site, ptm_type, wt[k], mut[k]])+'\n')
    

print(snp_count)

    # for k in wt:
    #     k = k.split('_')
    #     if k[0]==ptm_site  and '_'.join([k[1],k[2]])==ptm_type:
    #         prob = float(wt[k])


    # dat = {}
    # for k in wt:
    #     try:
    #         float(mut[k])# see if that's the mutate site
    #     except:
    #         print(k)
    #         continue
    #     if float(wt[k]) > 0.5:
    #         if float(mut[k])< 0.5:
    #             dat[k] = (float(wt[k]), float(mut[k]))
    #     if float(wt[k]) < 0.5:
    #         if float(mut[k])>0.5:
    #             dat[k] = (float(wt[k]), float(mut[k]))

    # new_dat = {}

    # h_thres = 0.5
    # l_thres = 0.5

    # for k in wt:
    #     site = int(k.split('_')[0])
    #     ptm = k.split('_')[1]+k.split('_')[2]
    #     if new_dat.get(ptm,-1)==-1:
    #         new_dat[ptm] = [0,0]
    #     if float(wt[k]) > 0.5:
    #         new_dat[ptm][0]+=1


    # for k in mut:
    #     site = int(k.split('_')[0])
    #     ptm = k.split('_')[1]+k.split('_')[2]
    #     if new_dat.get(ptm,-1)==-1:
    #         new_dat[ptm] = [0,0]
    #     if float(mut[k]) > 0.5:
    #         new_dat[ptm][1]+=1

    # with open('/local2/yuyan/PTM-Motif/Data/PTMVar/cardiac/res/'+uid+'_'+snp[0]+snp[1]+snp[2]+suffix+'_map.json','w') as f:
    #     json.dump(dat, f)


    # pprint(new_dat)

