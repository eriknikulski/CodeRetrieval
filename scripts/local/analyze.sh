#!/bin/bash

echo "RUNNING ANALYZE"

#python3.10 ../analyze.py --task=ngram
python3.10 ../analyze.py --type dataset --keep-duplicates
#python3.10 ../dataset_analytics.py
