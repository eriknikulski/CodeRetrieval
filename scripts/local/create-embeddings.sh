#!/bin/bash

echo "RUNNING FAIRSEQ MULTILINGUAL CREATION OF EMBEDDINGS ON CPU"

PROJ_DIR=".."
DATA_DIR=$PROJ_DIR"/save/data"

MODEL_CHECKPOINT=$PROJ_DIR/checkpoints/dual_encoder_decoder_lstm/checkpoint_best.pt

CODE_EMBEDDING_PATH=$DATA_DIR"/embeddings.pickle"

cat $DATA_DIR/with_url/fairseq.doc-code/train.code \
    | python3.10 $PROJ_DIR/fairseq/fairseq-create-embeddings \
      $DATA_DIR/fairseq.doc-code.doc-code \
      --task multilingual_embedding_creator \
      --user-dir $PROJ_DIR/fairseq/models \
      --lang-pairs doc-doc,code-doc,doc-code,code-code \
      --source-lang code --target-lang code \
      --path $MODEL_CHECKPOINT \
      --buffer-size 2000 --batch-size 1 \
      --beam 1 \
    > results.code-code.sys

echo "FINISHED EMBEDDING CREATION"