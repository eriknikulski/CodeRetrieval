#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --job-name=seq2seq
#SBATCH --mem=20G

echo "RUNNING ANALYZE"

module purge
module load modenv/hiera
module load GCCcore/11.3.0
module load Python/3.10.4

#python3.10 ../analyze.py --task=ngram
python3.10 ../analyze.py --type dataset
#python3.10 ../dataset_analytics.py
