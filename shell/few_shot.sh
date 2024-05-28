echo $1, $2, $3
CUDA_ID=$1
seed=$2
target_domain=$3  # movie, book
root_path="/your/root/path/TLRec/"
exp_name="alpaca_stage123_instrument"
train_data_path="$root_path/rec_data/$target_domain/train.json"
val_data_path="$root_path/rec_data/$target_domain/valid.json"
test_data_path="$root_path/rec_data/$target_domain/test.json"

echo "===== few-shot setting ====="
for sample in 16 64 256
do
    output_dir="$root_path/output_model/$target_domain/$exp_name/${seed}_${sample}/"
    mkdir -p $output_dir
    echo "seed: $seed, sample: $sample"
    CUDA_VISIBLE_DEVICES=$CUDA_ID python rec_tuning.py \
        --base_model "$root_path/base_model/llama-7b/" \
        --train_data_path $train_data_path \
        --val_data_path $val_data_path \
        --output_dir $output_dir \
        --resume_from_checkpoint "$root_path/instruction_model/$target_domain/$exp_name/" \
        --num_epochs 30 \
        --learning_rate 1e-4 \
        --sample $sample \
        --seed $seed

    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model "$root_path/base_model/llama-7b/" \
        --lora_weights "$root_path/output_model/$target_domain/$exp_name/${seed}_${sample}/" \
        --test_data_path $test_data_path
done
