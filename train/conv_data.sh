export raw_data=/home/daiyf/daiyf/PIXIU-train/pixiu_private-main/train/CRAdata/CRA_resample_0.045M.json
export conv_data=/home/daiyf/daiyf/PIXIU-train/pixiu_private-main/train/CRAdata/CRA_resample_0.045M_conv.json
export data_name=CRA
export dev_data=/home/daiyf/daiyf/PIXIU-train/pixiu_private-main/train/CRAdata/CRA-resample-dev3k.json
export train_data=/home/daiyf/daiyf/PIXIU-train/pixiu_private-main/train/CRAdata/CRA-resample-train4w.json

python scripts/convert_to_conv_data.py \
    --orig_data ${raw_data} \
    --write_data ${conv_data} \
    --dataset_name CRA
head -n 3000 ${conv_data} > ${dev_data}
tail -n +3001 ${conv_data} > ${train_data}