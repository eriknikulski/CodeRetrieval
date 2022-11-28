# Prerequisites
To create the necessary directory structure run ```python create_dir_structure.py```.

Download the ```java``` dataset from [https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip](https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip) and unzip it in the ```data/``` folder.

# Usage
## Synthetic
Run ```create_synth_sequences.py``` to create synthetic data.
For training run ```python train.py --data=synth --keep-duplicates```.

## Java
Run ```python preprocess.py``` with the necessary arguments, e.g. ```--data=java``` for java, to create a pandas dataframe that is later used in training.

To execute the training run ```python train.py --data=java --load-data```. For training with GPUs use the argument ```--gpu```. The created models will be saved in the folder ```save/model/```.

Run ```python infer.py``` to infer a few random elements from the ```java``` dataset using the previously created models. 