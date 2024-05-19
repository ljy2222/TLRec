echo $1
CUDA_ID=$1
test_data="/home/ljy/rec_data/book/test.json"

### step123 (0.5007100201638817)
##CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
##  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/step123/" \
##  --test_data_path $test_data
#
### step1 (0.4566020473457339)
##CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
##  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/step1/" \
##  --test_data_path $test_data
#
### step1_step12 (0.46631118438964525)
##CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
##  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/step1_step12/" \
##  --test_data_path $test_data
#
### step1_step12_step123 (0.452121491878475)
##CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
##  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/step1_step12_step123/" \
##  --test_data_path $test_data
#
### step123_instrument (0.6018251837069203)
##CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
##  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/step123_instrument/" \
##  --test_data_path $test_data
#
### step1_step12_step123_instrument (0.5379869425391565)
##CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
##  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/step1_step12_step123_instrument/" \
##  --test_data_path $test_data
#
## alpaca_step123
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/alpaca_step123/" \
#  --test_data_path $test_data
#
## alpaca_step123_instrument
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/alpaca_step123_instrument/" \
#  --test_data_path $test_data



## =====最新的实验结果=====
## llama base model (0.4620559005266933)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "" \
#  --test_data_path $test_data

## alpaca_step123 (0.6001888549983347)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home/ljy/TLRec/instruction_model/book/alpaca_step123/" \
#  --test_data_path $test_data

## alpaca_step123_instrument (0.6736764602229774)
#CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
#  --lora_weights "/home/ljy/TLRec/instruction_model/book/alpaca_step123_instrument/" \
#  --test_data_path $test_data

## alpaca_step123_instrument
##0_16 (0.6734121461473719)
##0_64 (0.6515310867903772)
##0_256 (0.)

0.6726851960623075
0.6514049761529968
0.666048321450141


0.6734121461473719
0.6563630244993236
0.6667856367930722

##
### alpaca_step123_instrument
### movie-book (0.5570085727592731)
### book-book (0.5434062448605597)
### 0.6066505567525509
### 0.6063575654635136
##CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
##  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/alpaca_step123_instrument/" \
##  --test_data_path $test_data
#
## step123+instrument
##0_16 (0.6117468085371027)
##0_64 (0.6219016516692903)
##0_256 (0.6348126168682824)