#!/usr/bin/env bash

export WANDB_API_KEY="INSERT_YOUR_WANDB_API_KEY"

## Uncomment the following line if you encounter
## distributed training problems (e.g., when you are using RTX4090 GPUs)
# export NCCL_P2P_DISABLE=1    

python -m BMI.main \
    --mode train \
    --model_info small \
    --data_root data/NQ320K/output \
    --data_subdir HKmI.bert-base-uncased.doc2query-t5-base-msmarco-ft_NQ320K \
    --decode_embedding 2 \
    --query_type realq ta genq docseg \
    --num_train_epochs 40 \
    --train_batch_size 128 --eval_batch_size 8 \
    --dropout_rate 0.1 --Rdrop 0.15 \
    --adaptor_layer_num 4 --adaptor_decode 1 --adaptor_efficient 1 \
    --output_vocab_size 30 \
    --max_input_length 64 \
    --max_output_length 10 \
    --logs_dir logs/NQ320K \
    --n_gpu 4 \
    --test1000 0 \
    --profiling 0 \