model_name_or_path=/media/data1/fengduanyu/llama-2-7b-chat-T/
ckpt_path=/home/daiyf/daiyf/PIXIU-train/CRA__model_2/model_14956


CUDA_VISIBLE_DEVICES=1 python src/entry_point/interface.py \
    --model_name_or_path $model_name_or_path \
    --ckpt_path $ckpt_path \
    --llama

