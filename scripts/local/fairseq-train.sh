#!/bin/bash

while getopts dpl: flag
do
    case "${flag}" in
        d) DIRTY=true;;
        p) PREPROCESS=true;;
        l) LANGUAGE=${OPTARG};;
    esac
done

echo "RUNNING FAIRSEQ MULTILINGUAL TRAINING ON CPU"

PROJ_DIR=".."
DATA_DIR=$PROJ_DIR"/save/data"

DIRTY_STR=""
DIRTY_CLI=""

if [ "$DIRTY" == true ]; then
    DIRTY_STR="dirty."
    DIRTY_CLI="--dirty"
    echo "RUNNING DIRTY..."
fi

if [ "$PREPROCESS" == true ]; then
  echo "CREATING DATA..."
  echo "CREATING DOC-DOC..."
  python3.10 $PROJ_DIR/create_fairseq_data.py --data doc $DIRTY_CLI
  python3.10 $PROJ_DIR/create_fairseq_data.py --data doc -u $DIRTY_CLI
  echo "CREATING CODE-CODE..."
  python3.10 $PROJ_DIR/create_fairseq_data.py --data code $DIRTY_CLI
  python3.10 $PROJ_DIR/create_fairseq_data.py --data code -u $DIRTY_CLI
  echo "CREATING DOC-CODE..."
  python3.10 $PROJ_DIR/create_fairseq_data.py --data doc code $DIRTY_CLI
  python3.10 $PROJ_DIR/create_fairseq_data.py --data doc code -u $DIRTY_CLI
  echo "CREATING CODE-CODE..."
  python3.10 $PROJ_DIR/create_fairseq_data.py --data code doc $DIRTY_CLI
  python3.10 $PROJ_DIR/create_fairseq_data.py --data code doc -u $DIRTY_CLI
  ## Preprocess/binarize the data
  echo "PREPROCESSING..."
  echo "PREPROCESSING DOC-DOC..."
  TEXT=$DATA_DIR/fairseq.${DIRTY_STR}doc-doc
  rm $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code/dict.doc.txt
  fairseq-preprocess --source-lang doc --target-lang doc \
      --trainpref $TEXT/train \
      --validpref $TEXT/valid \
      --testpref $TEXT/test \
      --destdir $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code \
      --workers 20

  echo "PREPROCESSING CODE-CODE..."
  TEXT=$DATA_DIR/fairseq.${DIRTY_STR}code-code
  rm $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code/dict.code.txt
  fairseq-preprocess --source-lang code --target-lang code \
      --trainpref $TEXT/train \
      --validpref $TEXT/valid \
      --testpref $TEXT/test \
      --destdir $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code \
      --workers 20

  echo "PREPROCESSING DOC-CODE..."
  TEXT=$DATA_DIR/fairseq.${DIRTY_STR}doc-code
  fairseq-preprocess --source-lang doc --target-lang code \
      --trainpref $TEXT/train \
      --validpref $TEXT/valid \
      --testpref $TEXT/test \
      --srcdict $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code/dict.doc.txt \
      --tgtdict $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code/dict.code.txt \
      --destdir $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code \
      --workers 20

  echo "PREPROCESSING CODE-DOC..."
  TEXT=$DATA_DIR/fairseq.${DIRTY_STR}code-doc
  fairseq-preprocess --source-lang code --target-lang doc \
      --trainpref $TEXT/train \
      --validpref $TEXT/valid \
      --testpref $TEXT/test \
      --srcdict $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code/dict.code.txt \
      --tgtdict $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code/dict.doc.txt \
      --destdir $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code \
      --workers 20
fi


mkdir -p $PROJ_DIR/checkpoints/dual_encoder_decoder_lstm/local/

if [ "$l" == "doc" ]; then
  LANGUAGE_PAIRS="doc-doc"
elif [ "$l" == "code" ]; then
  LANGUAGE_PAIRS="code-code"
else
  LANGUAGE_PAIRS="doc-doc,code-doc,doc-code,code-code"
fi

WANDB_PROJECT="seq2seq"

echo "TRAINING ${LANGUAGE_PAIRS}..."
fairseq-train \
    $DATA_DIR/fairseq.${DIRTY_STR}doc-code.doc-code \
    --task multilingual_translation \
    --lang-pairs ${LANGUAGE_PAIRS} \
    --user-dir $PROJ_DIR/fairseq/models \
    --arch dual_encoder_decoder_lstm \
    --optimizer sgd \
    --momentum 0.9 \
    --lr 0.01 \
    --batch-size 64 \
    --save-dir $PROJ_DIR/checkpoints/dual_encoder_decoder_lstm/local \
    --wandb-project $WANDB_PROJECT \
    --ddp-backend=legacy_ddp

echo "FINISHED TRAINING"