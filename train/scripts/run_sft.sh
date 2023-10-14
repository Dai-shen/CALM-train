#! /bin/bash
# llama1-7b --> abs_path:
# /home/daiyf/daiyf/HFmodel/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16
# llama2-7b-chat --> abs_path:
# /media/data1/fengduanyu/llama-2-7b-chat-T/
export CUDA_VISIBLE_DEVICES='0,1'  # Full
export WANDB_PROJECT=CRA-llama2-7b-chat
export WANDB_RUN_ID=CRA_0.045M
export WANDB_RESUME=allow
export ABS_PATH="/hy-tmp"
export PYTHONPATH="$ABS_PATH/pixiu_private-main/train"
export WANDB_SERVER_PORT=10086
export WANDB_API_KEY=8ec908e7dd12e9dc460578f6c82912fcd93d2b9d
model_name_or_path="/hy-tmp/pixiu_private-main/ourmodel/llama-2-7b-chat-T"

train_file="/hy-tmp/pixiu_private-main/train/CRAdata/CRA-resample-train4w.json"
validation_file="/hy-tmp/pixiu_private-main/train/CRAdata/CRA-resample-dev3k.json"
output_dir="$ABS_PATH/saved_models2/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir="/hy-tmp/hf_cache_dir_2"    # 代表缓存数据处理过程的路径
log_dir="/hy-tmp/train_log_dir_2"
mkdir -p ${cache_dir}
mkdir -p ${log_dir}
cutoff_len=2048  #  最长输入序列长度（LLaMA模型建议设置为1024以上，Bloom模型设置为512以上）
echo ${log_dir}

#FT
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --deepspeed configs/deepspeed_config.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --resume_from_checkpoint ...


#LoRA with 8bit
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora \
#     --use_int8_training \
#     --lora_config configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --resume_from_checkpoint ...

# LoRA without 8bit
nohup torchrun --nproc_per_node 2 src/entry_point/sft_train.py \
    --model_name_or_path ${model_name_or_path} \
    --bf16 True \
    --llama True \
    --use_lora True \
    --deepspeed configs/deepspeed_config_stage3.json \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    > ${log_dir}/train.log 2>&1 &
   # --fp16 \
   # --resume_from_checkpoint ...

# overwrite_output_dir 确保之前结果不需要，覆盖之前训练结果
# llama 默认是False
# nproc_per_node 2
# use_int8_training 代表采用8bit量化训练，可显著减少显存占用
# lora_config 给出了LoRA的参数配置。如果训练Bloom模型，则改为configs/lora_config_bloom.json
# deepspeed 训练的序列较长时，推荐使用deepspeed stage 3，能有效将模型参数分配到多卡上，留下空间加载更长的序列
# use_int8_training和deepspeed只能二选一，不可同时使用