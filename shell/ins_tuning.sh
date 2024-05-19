echo $1, $2, $3
CUDA_ID=$1
data_name=$2
instrument_name=$3

#echo "=====step123_instrument====="
#CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
#  --data_path "/home_nfs/haitao/data/ljy/CoTRec/instrument_data/$instrument_name/train.json" \
#  --output_dir "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step123_instrument/" \
#  --resume_from_checkpoint "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/$data_name/step123/"

echo "=====alpaca_step123_instrument====="
CUDA_VISIBLE_DEVICES=$CUDA_ID python ins_tuning.py \
  --train_data_path "/home/ljy/TLRec/instrument_data/$instrument_name/train.json" \
  --val_data_path "/home/ljy/rec_data/$data_name/valid.json" \
  --output_dir "/home/ljy/TLRec/instruction_model/$data_name/alpaca_step123_instrument/" \
  --resume_from_checkpoint "/home/ljy/TLRec/instruction_model/$data_name/alpaca_step123/"
CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
  --lora_weights "/home/ljy/TLRec/instruction_model/$data_name/alpaca_step123_instrument/" \
  --test_data_path "/home/ljy/rec_data/$data_name/test.json"