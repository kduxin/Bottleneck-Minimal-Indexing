#!/usr/bin/env bash

## CKPT path. Replace the following with your own path
## For example, CKPT="logs/NQ320K/BMI.bert-base-uncased.doc2query-t5-base-msmarco-ft_NQ320K.realq_genq_docseg/kary=30.realq-ta-genq-docseg.small.dev.recall.dem=2.ada=1.adaeff=1.adanum=4.RDrop=0.1-0.15-0_lre2.0d1.0_epoch=30-recall1=0.609631.ckpt"
CKPT=""

## Data path. Replace the following with your own path
DATA_ROOT="data/NQ320K/output"
DATA_SUBDIR="BMI.bert-base-uncased.doc2query-t5-base-msmarco-ft_NQ320K.realq_genq_docseg"

## Model size information. Replace the following with your own model size information
MODEL_INFO="small"

python -m BMI.main --mode eval \
--decode_embedding 2 \
--tree 1 \
--query_type \
--load_pretrained_encoder 0 \
--resume_from_checkpoint $CKPT \
--num_return_sequences 100 \
--model_info $MODEL_INFO \
--train_batch_size 32 --eval_batch_size 2 --dropout_rate 0.1 --Rdrop 0.1 \
--adaptor_decode 1 --adaptor_efficient 1 --adaptor_layer_num 4 \
--output_vocab_size 30 \
--max_input_length 64 \
--max_output_length 10 \
--data_root $DATA_ROOT \
--data_subdir $DATA_SUBDIR \
--profiling 0 \
--verbose 0 \
--test1000 0 \
--n_gpu 1 \