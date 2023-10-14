# model_name_or_path=/data/hanweiguang/Projects/Vicuna-LoRA-RLHF-PyTorch/weights/vicuna-7b
# model_name_or_path=decapoda-research/llama-7b-hf
model_name_or_path='/data/hanweiguang/Projects/Vicuna-LoRA-RLHF-PyTorch/weights/vicuna-7b'
# ckpt_path='/nfs/a100-80G-17/jiyunjie/finetuned_ckpt/on_belle_tokenizer50k_openinstr_zh/zh_alpaca_gpt3.5_gpt4_sharegpt_epoch=2-step=20652'
ckpt_path='/data/hanweiguang/Projects/Vicuna-LoRA-RLHF-PyTorch/weights/vicuna-7b'

# ft
# python ${ckpt_path}/zero_to_fp32.py \
#     ${ckpt_path} ${ckpt_path}/pytorch_model.bin

# CUDA_VISIBLE_DEVICES无效，要用localhost指定
# https://www.deepspeed.ai/getting-started/#resource-configuration-single-node
# deepspeed inference解码不支持beam search
# deepspeed inference可能会影响生成效果
# https://github.com/microsoft/DeepSpeed/issues/3452
# deepspeed --include localhost:1,5,6,7 src/interface.py \
#     --model_name_or_path $model_name_or_path \
#     --ckpt_path $ckpt_path \
#     --llama \

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# python src/interface.py \
#     --model_name_or_path $model_name_or_path \
#     --ckpt_path $ckpt_path \
#     --llama \

# lora
python src/entry_point/interface.py \
    --model_name_or_path $model_name_or_path \
    --ckpt_path $ckpt_path \
    --llama \
    --local_rank $1
    # --use_lora \
    
