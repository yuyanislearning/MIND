import json
from pprint import pprint

# SNPs = {'O94759':[('V','934','I')], 'P02649':[('C','130','R')], 'Q5S007':[('G','2019','S')],
#     'Q8TF76':[('V','76','E')],'Q96RQ3':[('H','464','P')],'Q9BSA9':[('M','393','T')],'Q9C0K1':[('A','391','T')]}
uid = 'Q5S007'
snps = [("M","712","V"),("R","793","M"),("Q","930","R"),("R","1067","Q"),("S","1096","C"),
        ("I","1122","V"),("S","1228","T"),("I","1371","V"),("R","1441","C"),("R","1441","G"),("R","1441","H"),
        ("R","1514","Q"),("P","1542","S"),("V","1598","E"),("Y","1699","C"),("R","1728","H"),("R","1728","L"),
        ("M","1869","T"),("R","1941","H"),("I","2012","T"),("G","2019","S"),("I","2020","T"),("T","2031","S"),
        ("T","2141","M"),("R","2143","H"),("D","2175","H"),("Y","2189","C"),("T","2356","I"),("G","2385","R"),
        ("V","2390","M"),("L","2439","I"),("L","2466","H")]
for snp in snps:#SNPs:
    with open('/workspace/PTM/Data/PTMVar/res/'+uid+'_'+snp[0]+snp[1]+snp[2]+'_OPTM.json') as f:
        mut = json.load(f)

    with open('/workspace/PTM/Data/PTMVar/res/'+uid+'_OPTM.json') as f:
        wt = json.load(f)

    dat = {}
    for k in wt:
        try:
            float(mut[k])# see if that's the mutate site
        except:
            print(k)
            continue
        if float(wt[k]) > 0.5:
            if float(mut[k])< 0.5:
                dat[k] = (float(wt[k]), float(mut[k]))
        if float(wt[k]) < 0.5:
            if float(mut[k])>0.5:
                dat[k] = (float(wt[k]), float(mut[k]))

    new_dat = {}

    h_thres = 0.5
    l_thres = 0.5

    for k in wt:
        site = int(k.split('_')[0])
        ptm = k.split('_')[1]+k.split('_')[2]
        if new_dat.get(ptm,-1)==-1:
            new_dat[ptm] = [0,0]
        if float(wt[k]) > 0.5:
            new_dat[ptm][0]+=1


    for k in mut:
        site = int(k.split('_')[0])
        ptm = k.split('_')[1]+k.split('_')[2]
        if new_dat.get(ptm,-1)==-1:
            new_dat[ptm] = [0,0]
        if float(mut[k]) > 0.5:
            new_dat[ptm][1]+=1

    with open('/workspace/PTM/Data/PTMVar/res/'+uid+'_'+snp[0]+snp[1]+snp[2]+'_OPTM_map.json','w') as f:
        json.dump(dat, f)


    pprint(new_dat)

