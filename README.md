# Prerequisites
Run ```pip install -r requirements.txt```.
To create the necessary directory structure run ```python create_dir_structure.py```.

Download the ```java``` dataset from [https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip](https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip) and unzip it in the ```data/``` folder.

# Usage
There are multiple bash scripts for preprocessing, training, embedding creation and retrieval.

## SLURM
If you are running this on slurm run ```cp scripts/slurm/* scripts/```.
Run ```chmod +x scripts/*.sh``` to enable the execution of the shell scripts.
In all the scripts you need to specify your account. 
To do this set your account in every occurence of ```#SBATCH -A YOUR_ACCOUNT```.
You also need to replace either the variable ```$NUMBER``` or ```$BIN_PATH``` to enable access to the fairseq cli tools.

## Local
If you are running this on slurm run ```cp scripts/slurm/* scripts/```.
Run ```chmod +x scripts/*.sh``` to enable the execution of the shell scripts.
You may experience performance issues.
To improve runtime, set the cli flag ```-d``` for the training script.
This reduces the preprocessing.
However, if you are not using GPUs the training will probably take forever.


## General
If you want to use WandB you need to set the variable ```WANDB_PROJECT``` to the name of your project in the training script.

### Preprocessing and Training
Run ```./scripts/fairseq-train.sh``` to run the dual Encoder-Decoder LSTM-based model.
Or run ```./scripts/fairseq-train-transformer.sh``` to run the dual Encoder-Decoder Transformer-based model.

Flags are:  
```-p```, this enables preprocessing and creates the datasets needed for training.  
```-d```, this skips major parts of preprocessing but still creates the datasets (only when ```-p``` set).  
```-l```, here you can specify the language pairs, if not set training is done on all the combinations of docstring and code. 

For ```-l doc``` the model is used as an autoencoder for docsting.
For ```-l code``` the model is used as an autoencoder for source code.


### Embedding Creation
Note that right now only source code embeddings are created.
If you want to change this, alter the script.
You also may want to change the variable ```$MODEL_CHECKPOINT``` in the script to match the model from training.

Run ```./scripts/fairseq-create-embeddings.sh``` to create the embeddings.

### Retrieval / Inference
Run ```./scripts/fairseq-infer.sh``` to run retrieval and evaluation on the queries defined in ```eval/prediction.csv```.
Detailed results are written in ```results.code-code.sys```. 
For evaluation these are processed and can be found in ```predictions.csv```.  

By default, the 100 closest element are returned. 
This can be changed by supplying the argument ```-c``` followed by a number. 
I.e. ```-c 10``` returns 10 results for every query.