# CALM: Credit and Risk Assessment Large Language Model

- Due to licensing restrictions on [LLaMA](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) weights, the model cannot be used for commercial purposes. Please adhere strictly to LLaMA's usage policy.
- Considering the limitations of LLaMA's license, we cannot directly distribute the complete model weights. Here, we are only releasing the LoRA weights of [CALM-7B](https://huggingface.co/daishen/CALM-7B).

## 1. Preparing the environment

Creating the environment using Conda, followed by installing the required packages using pip.

```bash
pip install -r requirements.txt
```

## 2. Run

### 2.1 Download data

Before running, please download [rawdata](https://huggingface.co/datasets/daishen/CALM-Data/tree/main) to `data/CRA_resample_0.045M.json`

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

We designate the first 3000 entries as the validation set, while the remaining data serves as the training set.

### 2.2 Model training

Training strategy

* LoRA + int8

The initiation script for training is written in `train/scripts/run.sh`. You will need to modify the parameters in `run.sh` according to your specific requirements.

```bash
bash scripts/run_sft.sh
```

- model_name_or_path: The pretrained model (if it is an [LLaMA](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model, it needs to be converted to the hf format beforehand in order to be loaded using from_pretrained)
- train_file: Training data
- validation_file: Validation data
- output_dir: Path to the training logs and model saves
- cache_dir: Path to the cache data processing process
- cutoff_len: Maximum input sequence length ([LLaMA](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model suggests setting it to 1024 or above, [Bloom](https://huggingface.co/bigscience/bloom) model suggests setting it to 512 or above)

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

* use_lora: Training using LoRA
* use_int8_training: Training with 8-bit quantization, which significantly reduces memory usage
* lora_config: The parameter configuration for LoRA is provided. If training a [Bloom](https://huggingface.co/bigscience/bloom) model, it should be changed to "`configs/lora_config_bloom.json`"
* deepspeed When training sequences are long, it is recommended to utilize deepspeed stage 3, which effectively distributes model parameters across multiple cards, allowing room to load even longer sequences

**Note:** Please be aware that you can only choose between "`use_int8_training`" and "`deepspeed`"; they cannot be used simultaneously.

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

The highest-level directory stores the final model obtained from the training process.

#### 2.2.2 Merge Model with LORA

If you wish to merge the weights of LoRA with a pre-trained model, you can execute the following command:

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

The merged weights will be saved in the "`output_path`" directory. You can subsequently load them directly using "`from_pretrained`".
