echo $1, $2, $3
CUDA_ID=$1
target_domain=$2  # movie, book
source_domain=$3  # netflix, amazon-book
root_path="/your/root/path/TLRec/"

echo "===== zero-shot setting ====="
CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
  --base_model "$root_path/base_model/llama-7b/" \
  --train_data_path "$root_path/cot_data/stage123.json" \
  --val_data_path "$root_path/rec_data/$target_domain/valid.json" \
  --output_dir "$root_path/instruction_model/$target_domain/alpaca_stage123/" \
  --resume_from_checkpoint "$root_path/instruction_model/$target_domain/alpaca/"
CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
  --base_model "$root_path/base_model/llama-7b/" \
  --train_data_path "$root_path/instruction_data/$source_domain/train.json" \
  --val_data_path "$root_path/rec_data/$target_domain/valid.json" \
  --output_dir "$root_path/instruction_model/$target_domain/alpaca_stage123_instrument/" \
  --resume_from_checkpoint "$root_path/instruction_model/$target_domain/alpaca_stage123/"
CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
  --base_model "$root_path/base_model/llama-7b/" \
  --lora_weights "$root_path/instruction_model/$target_domain/alpaca_stage123_instrument/" \
  --test_data_path "$root_path/rec_data/$target_domain/test.json"