# CALM训练代码

 | [English](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/README_en.md) | [中文](https://github.com/LianjiaTech/BELLE/blob/main/train/README.md)

## 1. 准备环境

通过conda创建环境，然后pip安装需要的包

```bash
pip install -r requirements.txt
```

## 2. Run

### 2.1 Download data

下载原始数据 [CRA_resample_0.045M](https://huggingface.co/datasets/daishen/CALM-Data/tree/main) 到 data 文件夹下

#### 2.1.1 Convert data format

```bash
export raw_data=/path_to/CRA_resample_0.045M.json
export conv_data=/path_to/CRA_resample_0.045M_conv.json
export data_name=CRA
export dev_data=/path_to/CRA-resample-dev3k.json
export train_data=/path_to/CRA-resample-train4w.json

python scripts/convert_to_conv_data.py \
    --orig_data ${raw_data} \
    --write_data ${conv_data} \
    --dataset_name CRA
head -n 3000 ${conv_data} > ${dev_data}
tail -n +3001 ${conv_data} > ${train_data}
```

我们选取前3000条作为验证集，其余数据作为训练集

### 2.2 模型训练

训练配置

* LoRA + int8

训练的启动脚本写在scripts/run.sh，你需要按照实际需求修改run.sh中的参数

```bash
bash scripts/run_sft.sh
```

- model_name_or_path 代表预训练模型（如果是LLaMA模型，需事先转为hf格式才能通过from_pretrained读取）
- train_file 代表训练数据
- validation_file 代表验证数据
- output_dir 代表训练日志和模型保存的路径
- cache_dir 代表缓存数据处理过程的路径
- cutoff_len 代表最长输入序列长度（LLaMA模型建议设置为1024以上，Bloom模型设置为512以上）

#### 2.2.1 LoRA

```bash
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
```

**参数说明**

* use_lora 代表采用LoRA训练
* use_int8_training 代表采用8bit量化训练，可显著减少显存占用
* lora_config 给出了LoRA的参数配置。如果训练Bloom模型，则改为configs/lora_config_bloom.json
* deepspeed 训练的序列较长时，推荐使用deepspeed stage 3，能有效将模型参数分配到多卡上，留下空间加载更长的序列

**注意**：use_int8_training和deepspeed只能二选一，不可同时使用

output_dir目录的文件结构如下：

```
output_dir/
├── checkpoint-244/
│   ├── pytorch_model.bin
│   └── trainer_state.json
├── checkpoint-527/
│   ├── pytorch_model.bin
│   └── trainer_state.json
├── adapter_model.bin
├── print_log.txt
└── adapter_config.json
```

最上级目录存储训练的最终模型

#### 2.2.2 合并LoRA权重

如果您想要实现LoRA权重与预训练模型的合并，可运行如下命令：

```bash
model_name_or_path=model_path_to/llama-2-7b-chat-T/
lora_path=lora_path_to/checkpoint_2/3739
output_path=out_path_to/CRA__model_2/model_3739

CUDA_VISIBLE_DEVICES=0 python src/merge_llama_with_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --output_path ${output_path} \
    --lora_path ${lora_path} \
    --llama
```

合并后的权重保存在output_path目录下，后续可通过from_pretrained直接加载
