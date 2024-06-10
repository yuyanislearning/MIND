<img src='header.png' width=20% height=20%>

# MIND-S AI4PTM
<img src='GA.png'>

MIND-S (Multilabel INterpretable Deep learning method for PTM prediction) is a deep learning tool for making PTM predictions based on protein sequence, or protein structure. MIND-S features interpretability of the model through evaluating the importance for each input residual to identify the important residual for making a prediction. MIND-S can also be utilized as a tool for evaluating effects of mutations (e.g. SNPs) that eventually affect protein sequence. By comparing the PTM predictions between wildtype and mutant protein, MIND-S can give hint on whether the mutation will affect PTMs.

## Set up
Install MIND:
```bash
git clone https://github.com/yuyanislearning/MIND.git
cd MIND
```

We suggest building environment via docker or conda or using platforms with tensorflow2 installed.

### Build environment from docker file (Optional)
We provide a Dockerfile which directly sets up the tensorflow and relevant package. More information about docker can be found [here](https://www.docker.com/). 
The prerequisite for using the docker can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow).
After prerequisite is satisfied, you can use the dockerfile [here](docker_build/Dockerfile)
Run the following to build a docker image
```bash
cd docker_build
```
```bash
mv [path to Downloaded dockerfile]/Dockerfile ./
```
You need to replace [path to Downloaded dockerfile] with your file path of the Dockerfile you downloaded. 
```bash
docker build -t yuyanislearning/mind:1.0 .
```
Then run the following to run a docker container
```bash
docker run --gpus all -it --rm -v [Path to your working directory, need to contain MIND]:/workspace yuyanislearning/mind:1.0
```

### Build environment with conda (Optional)
Follow the instruction to install [tensorflow2](https://www.tensorflow.org/install/pip).

Install required packages:
```bash
pip install -r requirement
```

## Make predictions
MIND allows batch predictions for multiple proteins. A fasta files contains all protein sequence can be used as the input with run the following code.
A json file with ptm information (uid_site_PtmType) and prediction scores will be return.
An example code using protein Q5S007 fasta sequence is shown below:
```bash
mkdir temp
python batch_predict.py \
  --pretrain_name saved_model/MIND_fifteenfold \
  --data_path  sample/Q5S007.fa\
  --res_path temp \
  --n_fold 15 
```


## Make interpretations
MIND supports interpretation for individual PTM prediction. The fasta file of the protein interested should be provided and the ptm site and ptm type should also be provided. A list of supported ptm types are list here:
'Hydro_K','Hydro_P','Methy_K','Methy_R','N6-ace_K','Palm_C','Phos_ST','Phos_Y','Pyro_Q','SUMO_K','Ubi_K','glyco_N','glyco_ST'.
The following example code will run the saliency analysis on Phosphorylation on site 203 of protein P04150, and return a figure of surronding saliency scores.

```bash
python predict_saliency.py \
  --inter \
  --pretrain_name saved_model/MIND_fifteenfold \
  --data_path sample/P04150.fa \
  --res_path temp \
  --site 203 \
  --ptm_type Phos_ST
```

## examine SNP effect
<img src='SNP.png'>
MIND supports examination of SNP effect on PTMs. The fasta file of the protein interested and the SNP information should be provided. 
The following code will provide the results of both wild type and mutant protein sequence. 
Please change the code in '[]' to your input.

```bash
python PTMSNP.py \
  --pretrain_name saved_model/MIND_fifteenfold \
  --data_path [path to fasta file] \
  --res_path [path to store result] \
  --snp [snp e.g. R_1022_C] \
  --n_fold 15
```

## Cite
Please cite the following article for usage. 

__MIND-S is a deep-learning prediction model for elucidating protein post-translational modifications in human diseases.__

[Cell Reports Methods, 2023](https://doi.org/10.1016/j.crmeth.2023.100430)


## Contact Email
yuyan666@g.ucla.edu
