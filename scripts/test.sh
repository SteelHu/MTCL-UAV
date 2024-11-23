#!/bin/bash
# 实验目的：测试专用
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# 设置模型和数据路径
model_name=PathFormer
root_path_name=./dataset/ALFA_dataset1
model_id_name=dataset1_test
data_name=ALFA_ad
anomaly_ratio=35
d_model=16
seq_len=96
learning_rate=0.0001
train_epochs=50
patience=30
batch_norm=1
batch_size=128
residual_connection=1


# 运行 Python 脚本并将输出重定向到日志文件
python -u run.py \
--is_training 1 \
--root_path $root_path_name \
--model_id $model_id_name \
--model $model_name \
--data $data_name \
--loss_save_name "{model_id_name}.csv" \
--features M \
--seq_len $seq_len \
--pred_len $seq_len \
--num_nodes 18 \
--layer_nums 3 \
--batch_norm $batch_norm \
--residual_connection $residual_connection \
--k 3 \
--d_model $d_model \
--d_ff $d_model \
--train_epochs $train_epochs \
--patience $patience \
--lradj 'TST' \
--itr 1 \
--anomaly_ratio $anomaly_ratio \
--batch_size $batch_size \
--learning_rate $learning_rate
