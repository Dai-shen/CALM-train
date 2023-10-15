#! /bin/bash
# llama1-7b --> abs_path:
# /home/daiyf/daiyf/HFmodel/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16
# llama2-7b-chat --> abs_path:
# /media/data1/fengduanyu/llama-2-7b-chat-T/
export CUDA_VISIBLE_DEVICES='0,1,2,3'  # Full
export WANDB_PROJECT=CRA-llama2-7b-chat
export WANDB_RUN_ID=CRA_0.045M
export WANDB_RESUME=allow
export ABS_PATH=""
export PYTHONPATH="path_to/train"
export WANDB_SERVER_PORT=10086
export WANDB_API_KEY=WANDB_API_KEY
model_name_or_path="path_to/llama-2-7b-chat-T"

train_file="path_to/CRA-resample-train4w.json"
validation_file="path_to/CRA-resample-dev3k.json"
output_dir="$ABS_PATH/saved_models2/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir="path_to/hf_cache_dir_2"    # 代表缓存数据处理过程的路径
log_dir="/path_to/train_log_dir_2"
mkdir -p ${cache_dir}
mkdir -p ${log_dir}
cutoff_len=2048  #  最长输入序列长度（LLaMA模型建议设置为1024以上，Bloom模型设置为512以上）
echo ${log_dir}

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