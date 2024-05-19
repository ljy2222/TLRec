# TLRec: A Transfer Learning Framework to Enhance Large Language Models for Sequential Recommendation Tasks

## 1. Introduction
This is the implementation code of TLRec, 
a novel transfer learning framework aimed at enhancing LLMs for sequential recommendation tasks.

## 2. Preparation
### 2.1 Requirements
- conda create -n llm4rec python==3.10
- conda activate llm4rec
- pip install -r requirements.txt
- pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
- python -m bitsandbytes
- pip uninstall peft -y
- pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08

### 2.2 Data Preparation
To process CoT-based data, just use:
```
python xxxx
```
To process rec-tuning data, just use:
```
python xxxx
```

## 3. TLRec Training and Evaluation
To execute DAMCAR training, just use
```
# ===== movie =====
nohup bash ./shell/cot_tuning.sh 0 movie > ./nohup_logs/cot_tuning_movie.txt 2>&1 &
nohup bash ./shell/ins_tuning.sh 0 movie netflix > ./nohup_logs/ins_tuning_movie_netflix.txt 2>&1 &
nohup bash ./shell/rec_tuning.sh 0 0 movie alpaca_step123_instrument > ./nohup_logs/rec_tuning_0_movie_alpaca_step123_instrument.txt 2>&1 &

# ===== book =====
nohup bash ./shell/cot_tuning.sh 1 book > ./nohup_logs/cot_tuning_book.txt 2>&1 &
nohup bash ./shell/ins_tuning.sh 1 book amazon-book > ./nohup_logs/ins_tuning_book_amazon_book.txt 2>&1 &
nohup bash ./shell/rec_tuning.sh 1 0 book alpaca_step123_instrument > ./nohup_logs/rec_tuning_0_book_alpaca_step123_instrument.txt 2>&1 &
```
To execute DAMCAR evaluation, just use
```
# ===== movie =====
nohup bash ./shell/evaluate_movie.sh 7 > ./nohup_logs/evaluate_movie.txt 2>&1 &

# ===== book =====
nohup bash ./shell/evaluate_book.sh 7 > ./nohup_logs/evaluate_book.txt 2>&1 &
```

## 4. Acknowledgements
Our code is based on the implementation of [TALLRec](https://github.com/SAI990323/TALLRec) and [NIR](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec). Thanks their awesome works.

# 下列内容请忽略，仅用于测试
# ========== 实验评估 ==========
# ===== movie =====
CUDA_VISIBLE_DEVICES=7 python evaluate.py \
  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/movie/alpaca_step123_instrument/" \
  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/movie/test.json"

CUDA_VISIBLE_DEVICES=7 python evaluate.py \
  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/alpaca_step123_instrument/0_16/" \
  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/movie/test.json"

CUDA_VISIBLE_DEVICES=7 python evaluate.py \
  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/alpaca_step123_instrument/0_64/" \
  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/movie/test.json"

CUDA_VISIBLE_DEVICES=7 python evaluate.py \
  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/movie/alpaca_step123_instrument/0_256/" \
  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/movie/test.json"

# ===== book =====
CUDA_VISIBLE_DEVICES=5 python evaluate.py \
  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/instruction_model/book/alpaca_step123_instrument/" \
  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/book/test.json"

CUDA_VISIBLE_DEVICES=5 python evaluate.py \
  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/book/alpaca_step123_instrument/0_16/" \
  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/book/test.json"

CUDA_VISIBLE_DEVICES=6 python evaluate.py \
  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/book/alpaca_step123_instrument/0_64/" \
  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/book/test.json"

CUDA_VISIBLE_DEVICES=7 python evaluate.py \
  --lora_weights "/home_nfs/haitao/data/ljy/CoTRec/output_model/book/alpaca_step123_instrument/0_256/" \
  --test_data_path "/home_nfs/haitao/data/ljy/rec_data/book/test.json"