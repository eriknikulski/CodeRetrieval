#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --job-name=seq2seq
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4

echo "RUNNING FAIRSEQ INFERENCE ON GPU"

RETRIEVAL_COUNT=100

while getopts c: flag
do
    case "${flag}" in
        c) RETRIEVAL_COUNT=${OPTARG};;
    esac
done

PROJ_DIR=".."
DATA_DIR=$PROJ_DIR"/save/data"

MODEL_CHECKPOINT=$PROJ_DIR/checkpoints/dual_encoder_decoder_lstm/checkpoint_best.pt

module purge
module switch modenv/hiera
module load GCCcore/11.3.0
module load CUDA/11.8.0
module load Python/3.10.4

CODE_EMBEDDING_PATH=$DATA_DIR"/embeddings.pickle"
OUT_PATH=$PROJ_DIR"/out.out"

FAIRSEQ_OUT=$PROJ_DIR"/results.code-code.sys"
PRED_FILE=$PROJ_DIR"/predictions.csv"
QUERY_FILE=$PROJ_DIR"/eval/queries.csv"
ANNOTATION_FILE=$PROJ_DIR"/eval/annotationStore.csv"

tail --lines=+2 $QUERY_FILE \
    | python3.10 $PROJ_DIR/fairseq/fairseq-infer \
      $DATA_DIR/fairseq.doc-code.doc-code \
      --task multilingual_retrieval \
      --user-dir $PROJ_DIR/fairseq/models \
      --lang-pairs doc-doc,code-doc,doc-code,code-code \
      --source-lang doc --target-lang code \
      --path $MODEL_CHECKPOINT \
      --buffer-size 2000 --batch-size 1 \
      --beam 1 \
      --nbest $RETRIEVAL_COUNT \
    > $FAIRSEQ_OUT

python3.10 $PROJ_DIR/convert_results.py --input $FAIRSEQ_OUT --output $PRED_FILE

python3.10 $PROJ_DIR/eval/relevanceeval.py $ANNOTATION_FILE $PRED_FILE

echo "FINISHED INFERENCE"