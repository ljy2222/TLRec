echo $1, $2, $3, $4
CUDA_ID=$1
seed=$2
data_name=$3
exp_name=$4
root="/home/ljy/TLRec/"  # 根路径
train_data="/home/ljy/rec_data/$data_name/train.json"  # 训练集
val_data="/home/ljy/rec_data/$data_name/valid.json"  # 验证集
#val_data="/home/ljy/rec_data/$data_name/test.json"

# 基础模型
base_model="/home/ljy/llm_models/llama-7b/"  # modelscope的llama

# 加载lora权重
instruction_model="$root/instruction_model/$data_name/$exp_name/"  # alpaca

for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 16 64 256
        do
            output_dir="$root/output_model/$data_name/$exp_name/${seed}_${sample}/"  # 保存训练后的lora权重
            mkdir -p $output_dir
            echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
            CUDA_VISIBLE_DEVICES=$CUDA_ID python -u rec_tuning.py \
                --base_model $base_model \
                --train_data_path $train_data \
                --val_data_path $val_data \
                --output_dir $output_dir \
                --batch_size 128 \
                --micro_batch_size 32 \
                --num_epochs 30 \
                --learning_rate $lr \
                --cutoff_len 512 \
                --lora_r 8 \
                --lora_alpha 16\
                --lora_dropout $dropout \
                --lora_target_modules '[q_proj,v_proj]' \
                --train_on_inputs \
                --group_by_length \
                --resume_from_checkpoint $instruction_model \
                --sample $sample \
                --seed $seed

            CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
                --lora_weights "$root/output_model/$data_name/$exp_name/${seed}_${sample}/" \
                --test_data_path "/home/ljy/rec_data/$data_name/test.json"
        done
    done
done