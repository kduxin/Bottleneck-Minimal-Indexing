#!/usr/bin/env bash

export WANDB_API_KEY="4cd046a5edd8dd0a58a2060aa5177f4639871e60"

## Uncomment the following line if you encounter
## distributed training problems (e.g., when you are using RTX4090 GPUs)
# export NCCL_P2P_DISABLE=1    

python -m NCIRetriever.main \
    --mode train \
    --model_info small \
    --data_root data/Marco-Lite/output \
    --data_subdir BMI.bert-base-uncased.doc2query-t5-base-msmarco.genq \
    --decode_embedding 2 \
    --query_type realq ta genq docseg \
    --num_train_epochs 40 \
    --train_batch_size 64 --eval_batch_size 2 \
    --dropout_rate 0.1 --Rdrop 0.15 \
    --adaptor_layer_num 4 --adaptor_decode 1 --adaptor_efficient 1 \
    --output_vocab_size 30 \
    --max_input_length 64 \
    --max_output_length 10 \
    --logs_dir logs/Marco-Lite \
    --n_gpu 4 \
    --test1000 0 \
    --profiling 0 \
