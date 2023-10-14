#! /bin/bash
# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='0,1'
export WANDB_PROJECT=debug
export WANDB_RUN_ID=debug
export WANDB_RESUME=allow
export PYTHONPATH='/data/hanweiguang/Projects/BELLE/train'

# model_name_or_path="decapoda-research/llama-7b-hf"
model_name_or_path="bigscience/bloomz-560m"

train_file=/data/hanweiguang/Projects/BELLE/data/test_data/test_pt.jsonl
validation_file=/data/hanweiguang/Projects/BELLE/data/test_data/test_pt.jsonl
output_dir=/data/hanweiguang/Projects/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}
rm -rf $output_dir
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=32


# FT
torchrun --nproc_per_node 2 src/entry_point/pt_train.py \
    --model_name_or_path ${model_name_or_path} \
    --deepspeed configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 1e-7 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --report_to "tensorboard" \
    --bf16 \
    # --fp16 \
    # --llama

# debug lora single node multiple gpus
# torchrun --nproc_per_node 2 --rdzv-endpoint "127.0.0.1:30012" \
#     "src/entry_point/pt_train.py" \
#     --model_name_or_path ${model_name_or_path} \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 1 \
#     --model_max_length ${cutoff_len} \
#     --learning_rate 3e-4 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --save_strategy "steps" \
#     --save_total_limit 1 \
#     --evaluation_strategy "steps" \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#     --report_to tensorboard \
#     --use_lora \
#     --lora_config configs/lora_config_bloom.json \
#     --deepspeed configs/deepspeed_config_stage3.json \
#     --bf16 \
    # --fp16 \
    # --use_int8_training \
#     # --resume_from_checkpoint "/data/hanweiguang/Projects/BELLE/saved_models/boxue_debug_debug/checkpoint-5" \
#     # --llama \

# master_addr='10.201.102.66'
# master_port='65530'

# # debug lora multiple nodes
# torchrun \
#     --nproc_per_node 1 \
#     --nnode 2 \
#     --node_rank $1 \
#     --master_addr $master_addr \
#     --master_port $master_port \
#     'src/entry_point/pt_train.py' \
#     --model_name_or_path ${model_name_or_path} \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 40 \
#     --model_max_length ${cutoff_len} \
#     --learning_rate 3e-4 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --evaluation_strategy "steps" \
#     --fp16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#     --deepspeed configs/deepspeed_config_stage3.json \
#     --lora_config configs/lora_config_bloom.json \
#     --use_lora \
#     # --llama \