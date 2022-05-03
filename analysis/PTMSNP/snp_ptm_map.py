import json
from pprint import pprint

uid = 'Q5S007'
snps = [("R","793","M"),("I","1122","V"),("S","1228","T"),("M","712","V"),("R","1723","P"),
("R","1723","P"),("G","2019","S"),("I","2020","T"),("I","952","T"),("R","1628","P"),("G","2019","S"),
("I","2020","T"),("Q","930","R"),("I","1122","V"),("R","1398","H"),("R","1441","C"),("R","1441","G"),
("R","1441","H"),("M","1646","T"),("S","1647","T"),("N","2261","I"),("A","419","V"),("S","1283","T"),
("Q","930","R"),("R","1398","H"),("R","1441","C"),("R","1441","C"),("R","1441","G"),("R","1441","G"),
("R","1441","H"),("R","1441","H"),("T","2031","S"),("K","1359","I"),("I","1371","V"),("T","2031","S"),
("T","2031","S"),("I","952","T"),("Q","930","R")]

for snp in snps:
    with open('/workspace/PTM/Data/PTMVar/res/'+uid+'_'+snp[0]+snp[1]+snp[2]+'.json') as f:
        mut = json.load(f)

    with open('/workspace/PTM/Data/PTMVar/res/'+uid+'.json') as f:
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

    with open('/workspace/PTM/Data/PTMVar/res/'+uid+'_'+snp[0]+snp[1]+snp[2]+'_map.json','w') as f:
        json.dump(dat, f)


    pprint(new_dat)

