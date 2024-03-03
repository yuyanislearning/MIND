MIND-S with structure information for downstream application is in implementation. Here is an example on how to include structure for batch prediction.

### Construct adjcent matrix of amino acids from structure.
pdb2cont_map.py to convert the structure files to numpy format adjcent matrix. Call help function for more information.
```bash
python pdb2cont_map.py -h
```
Note: file of structure should be XXX-uniprotID-XXX.cif e.g., AF-A0A023FFB5-F1-model_v3.cif

### batch prediction
```bash
mkdir temp
mkdir out
python batch_predict.py \
  --pretrain_name saved_model/g_fifteenfold \
  --data_path  sample/Q5S007.fa \
  --adj_path sample_adj \ 
  --res_path out \
  --n_fold 15 \
  --graph
```
Note that this is similar to batch prediction without structure. This script will load the model trained on structure and require path to adjcent matrix calculated in the last step.