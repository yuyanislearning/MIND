import numpy as np
import os
import pdb

a = os.popen('ls --quoting-style=escape -U temp/')

for fle in a:
    fle = fle.strip()
    if '_AS.npy' in fle:
        try:
            a = np.load('temp/'+fle, allow_pickle=True)
        except:
            os.system('rm temp/'+ fle)
        if a.shape[0]<1:
            print(fle)
            os.system('rm temp/'+ fle)
            

