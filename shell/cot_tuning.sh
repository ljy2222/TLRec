echo $1, $2
CUDA_ID=$1
data_name=$2

#echo "=====step123====="
#CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
#  --data_path "/home_nfs/haitao/data/ljy/CoTRec/cot_data/step123.json" \
#  --output_dir "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step123/" \
#  --resume_from_checkpoint None

echo "=====alpaca_step123====="
CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
  --train_data_path "/home/ljy/TLRec/cot_data/step123.json" \
  --val_data_path "/home/ljy/rec_data/$data_name/valid.json" \
  --output_dir "/home/ljy/TLRec/instruction_model/$data_name/alpaca_step123/" \
  --resume_from_checkpoint "/home/ljy/TLRec/instruction_model/$data_name/alpaca/"
CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
  --lora_weights "/home/ljy/TLRec/instruction_model/$data_name/alpaca_step123/" \
  --test_data_path "/home/ljy/rec_data/$data_name/test.json"

#echo "=====step1====="
#CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
#  --data_path "/home_nfs/haitao/data/ljy/CoTRec/cot_data/step1.json" \
#  --output_dir "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step1/" \
#  --resume_from_checkpoint None \
#  --num_epochs 1 \
#  --learning_rate 1e-3

#echo "=====step1_step12====="
#CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
#  --data_path "/home_nfs/haitao/data/ljy/CoTRec/cot_data/step12.json" \
#  --output_dir "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step1_step12/" \
#  --resume_from_checkpoint "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step1/" \
#  --num_epochs 2 \
#  --learning_rate 3e-4

#echo "=====step1_step12_step123====="
#CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
#  --data_path "/home_nfs/haitao/data/ljy/CoTRec/cot_data/step123.json" \
#  --output_dir "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step1_step12_step123/" \
#  --resume_from_checkpoint "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step1_step12/" \
#  --num_epochs 3 \
#  --learning_rate 1e-4

## step1_step12_step123
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step1_step12_step123/" \
#  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/$data_name/test.json"