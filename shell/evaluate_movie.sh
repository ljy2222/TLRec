echo $1
CUDA_ID=$1
test_data="/home/ljy/rec_data/movie/test.json"

## step123 (0.4741569562146893)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/movie/step123/" \
#  --test_data_path $test_data
#
## step1 (0.3784367777349769)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/movie/step1/" \
#  --test_data_path $test_data
#
## step1_step12 (0.3857396796353364)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/movie/step1_step12/" \
#  --test_data_path $test_data
#
## step1_step12_step123 (0.4234078068823831)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/movie/step1_step12_step123/" \
#  --test_data_path $test_data
#
## step123_instrument (0.6891592032614279)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/movie/step123_instrument/" \
#  --test_data_path $test_data
#
## step1_step12_step123_instrument (0.6710925462249615)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/movie/step1_step12_step123_instrument/" \
#  --test_data_path $test_data
#
# ============== rec_tunning ==============
## 0_16 (0.683198510529019)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/step1_step12_step123_instrument/0_16/" \
#  --test_data_path $test_data
#
## 0_64 (0.6809614952491012)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/step1_step12_step123_instrument/0_64/" \
#  --test_data_path $test_data
#
## 0_256 (0.6815453261427837)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/step1_step12_step123_instrument/0_256/" \
#  --test_data_path $test_data
#
## 0_16 (0.6965523882896764)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/step123_instrument/0_16/" \
#  --test_data_path $test_data
#
## 0_64 (0.6676838565742168)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/step123_instrument/0_64/" \
#  --test_data_path $test_data
#
## 0_256 (0.7226903569594247)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/step123_instrument/0_256/" \
#  --test_data_path $test_data
#





## =====最新的实验结果=====
## llama base model (0.3801361068310221)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "" \
#  --test_data_path $test_data

## alpaca_step123 (0.5627688430919363)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home/ljy/TLRec/instruction_model/movie/alpaca_step123/" \
#  --test_data_path $test_data

## alpaca_step123_instrument (0.7216029468412943)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home/ljy/TLRec/instruction_model/movie/alpaca_step123_instrument/" \
#  --test_data_path $test_data

# alpaca_step123_instrument
#0_16 (0.740076881099127)
#0_64 (0.7427111421417566)
#0_256 (0.7372660663841808)

0.7395973773754494
0.7425686954288649
0.7362709456856703

0.7443242006933743
0.7445007543656909
0.7509168753210067




# 论文中的结果
## alpaca_step123_instrument (0.7059257832562917)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/movie/alpaca_step123_instrument/" \
#  --test_data_path $test_data
# step123+instrument
#0_16 (0.7262294555726758)
#0_64 (0.7303644228299949)
#0_256 (0.7411000898818695)