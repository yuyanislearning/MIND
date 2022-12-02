<img src='header.png' width=20% height=20%>

# MIND-S AI4PTM
<img src='GA.png'>

MIND (Multilabel INterpretable Deep learning method for PTM prediction) is a deep learning tool for making PTM predictions based on protein sequence, or protein structure. MIND-S features interpretability through evaluating the saliency for each input residual. MIND can also be utilized as a tool for evaluating effects of genetic mutation (e.g. SNP) on PTM.

## Set up
Install MIND:
```bash
git clone https://github.com/yuyanislearning/MIND.git
cd MIND
```

### Build environment from docker file (Optional)
We provide a Dockerfile which directly sets up the tensorflow and relevant package. More information about docker can be found [here](https://www.docker.com/). 
The prerequisite for using the docker can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow).
After prerequisite is satisfied, you can download the dockerfile [here](https://drive.google.com/file/d/1SlUNKthEDH_RuTkDI02g8bmZa0lig7BL/view?usp=sharing)
Run the following to build a docker image
```bash
mkdir docker_build
cd docker_build
mv [path to Downloaded dockerfile]/Dockerfile ./
docker build -t yuyanislearning/mind:1.0 .
```
Then run the following to run a docker container
```bash
docker run --gpus all -it --rm -v [Path to your working directory, need to contain MIND]:/workspace yuyanislearning/mind:1.0
```


## Make predictions
MIND allows batch predictions for multiple proteins. A fasta files contains all protein sequence can be used as the input with run the following code.
A json file with ptm information (uid_site_PtmType) and prediction scores will be return.
An example code is shown below:
```bash
python batch_predict.py \
  --pretrain_name saved_model/MIND_fifteenfold \
  --data_path [path to fasta file] \
  --res_path [path to store result] \
  --n_fold 15 
```


## Make interpretations
MIND supports interpretation for individual PTM prediction. The fasta file of the protein interested should be provided and the ptm site and ptm type should also be provided. A list of supported ptm types are list here:
'Hydro_K','Hydro_P','Methy_K','Methy_R','N6-ace_K','Palm_C','Phos_ST','Phos_Y','Pyro_Q','SUMO_K','Ubi_K','glyco_N','glyco_ST'.
The following code will run the saliency analysis and return a figure of surronding saliency scores.

```bash
python predict_saliency.py \
  --inter \
  --pretrain_name saved_model/MIND_fifteenfold \
  --data_path [path to fasta file] \
  --res_path [path to store result] \
  --site [site of PTM] \
  --ptm_type [PTM type] 
```

## examine SNP effect
<img src='SNP.png'>
MIND supports examination of SNP effect on PTMs. The fasta file of the protein interested and the SNP information should be provided. 
THe following code will provide the results of both wild type and mutant protein sequence. 

```bash
python PTMSNP.py \
  --pretrain_name saved_model/MIND_fifteenfold \
  --data_path [path to fasta file] \
  --res_path [path to store result] \
  --snp [snp e.g. R_1022_C] \
  --n_fold 15
```

## Cite
