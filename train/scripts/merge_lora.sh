#! /bin/bash

model_name_or_path=/media/data1/fengduanyu/llama-2-7b-chat-T/
lora_path=/home/daiyf/daiyf/PIXIU-train/checkpoint_2/3739
output_path=/home/daiyf/daiyf/PIXIU-train/CRA__model_2/model_3739

CUDA_VISIBLE_DEVICES=0 python src/merge_llama_with_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --output_path ${output_path} \
    --lora_path ${lora_path} \
    --llama