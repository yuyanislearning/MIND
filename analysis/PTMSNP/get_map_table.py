import json
import pdb

SNPs = {'P51530':[('R','198','H')],'O75489':[('R','199','W')],'O00217':[('R','102','H')],
    'Q9UMS0':[('G','190','R'),('G','189','R')],'Q8TB37':[('D','105','Y')],'Q9P2R7':[('R','284','C')],
    'P15056':[('F','468','S'),('K','601','Q'),('T','599','R'),('K','601','Q')],
    'P02545':[('R','60','G'),('R','435','C'),('I','210','S')],
    'Q14896':[('P','608','L'),('Q','998','R'),('R','272','C')],'P12883':[('G','425','R')],'P10916':[('E','22','K')],
    'Q86TC9':[('R','955','W'),('P','961','L')],'P04049':[('A','237','T')],'O75792':[('R','186','W')],
    'P19429':[('R','162','P'),('S','166','F'),('R','162','Q')],'P09493':[('E','40','K')],
    'Q9P2R6':[('P','1262','R')],'Q01484':[('R','3906','W')],'Q13936':[('E','477','K')],
    'Q12809':[('P','251','S'),('S','320','L'),('D','323','N'),('R','1033','W')],'Q14524':[('R','569','W')]}


OPTM=True
if OPTM:
    suffix='_OPTM'
else:
    suffix=''
# SNPs = {'O94759':[('V','934','I')], 'P02649':[('C','130','R')], 'Q5S007':[('G','2019','S')],
#     'Q8TF76':[('V','76','E')],'Q96RQ3':[('H','464','P')],'Q9BSA9':[('M','393','T')],'Q9C0K1':[('A','391','T')]}
# uid = 'Q14896'
# snps = [("Q","998","R"),("P","608","L"),("R","272","C")]
for suffix in ['', '_OPTM']:
    fwi = open('/local2/yuyan/PTM-Motif/Data/PTMVar/res/cardiac_snp_interfere'+suffix+'.tsv','w')
    fwp = open('/local2/yuyan/PTM-Motif/Data/PTMVar/res/cardiac_snp_promote'+suffix+'.tsv','w')
    for uid in SNPs:#SNPs:
        for snp in SNPs[uid]:
            try:
                with open('/local2/yuyan/PTM-Motif/Data/PTMVar/res/'+uid+'_'+snp[0]+snp[1]+snp[2]+suffix+'_map.json') as f:
                    dat = json.load(f)
            except:
                print('/local2/yuyan/PTM-Motif/Data/PTMVar/res/'+uid+'_'+snp[0]+snp[1]+snp[2]+suffix+'_map.json')
                continue
            if len(dat)>0:
                for k in dat:
                    ks = k.split('_')
                    site = ks[0]
                    ptm = '_'.join(ks[1:3])
                    if dat[k][0]>dat[k][1]:
                        fwi.write('\t'.join([uid, snp[0], snp[1], snp[2], site, ptm, str(dat[k][0]),str(dat[k][1])])+'\n')
                    else:
                        fwp.write('\t'.join([uid, snp[0], snp[1], snp[2], site, ptm, str(dat[k][0]),str(dat[k][1])])+'\n')
    fwi.close()
    fwp.close()