# TLRec: A Transfer Learning Framework to Enhance Large Language Models for Sequential Recommendation Tasks

## 1. Introduction
This is the implementation code of TLRec, a novel transfer learning framework aimed at enhancing the cross-domain generalization of LLMs for sequential recommendation tasks.

## 2. Preparation
### 2.1 Requirements
- cuda 12.2
- python 3.10.0
- pytorch 2.0.1
- numpy 1.26.4
- pandas 2.2.2
- peft 0.3.0.dev0
- transformers 4.28.0
- sentencepiece 0.1.99
- scikit-learn 1.4.0
- accelerate 0.26.1
- loralib 0.1.2
- bitsandbytes 0.42.0
- datasets 2.16.1
- openai 1.12.0

### 2.2 Data Preparation
Before running the code, the working directory is expected to be organized as follows:
<details><summary>TLRec/</summary>
<ul>
    <li>cot_data/</li>
    <ul>
        <li>generate_data.py</li>
        <li>preprocess_data.py</li>
        <li>stage123.json</li>
    </ul>
    <li>instruction_data/</li>
    <ul>
        <li>netflix</li>
        <ul>
            <li>train.json</li>
        </ul>
        <li>amazon-book</li>
        <ul>
            <li>train.json</li>
        </ul>
    </ul>
    <li>rec_data/</li>
    <ul>
        <li>movie</li>
        <ul>
            <li>train.json</li>
            <li>valid.json</li>
            <li>test.json</li>
        </ul>
        <li>book</li>
        <ul>
            <li>train.json</li>
            <li>valid.json</li>
            <li>test.json</li>
        </ul>
    </ul>
    <li>shell/</li>
    <ul>
        <li>generate_cot_data.sh</li>
        <li>zero_shot.sh</li>
        <li>few_shot.sh</li>
    </ul>
    <li>base_model/</li>
    <li>instruction_model/</li>
    <li>output_model/</li>
    <li>ins_tuning.py</li>
    <li>rec_tuning.py</li>
    <li>evaluate.py</li>
</ul>
</details>

The datasets Movie and Book are available in ./rec_data, divided into training, validation, and test sets in a ratio of 8:1:1. 
They are saved as train.json, valid.json, and test.json, respectively. 

For TLRec, we introduce Netflix and Amazon-Book datasets as third-party scenarios. To generate CoT-based data for model training, please use:
```
bash ./shell/generate_cot_data.sh
```

## 3. TLRec Training and Evaluation
We execute instruction tuning to augment LLMs’ performance for sequential recommendation tasks under the zero-shot and few-shot settings. 
For TLRec training and evaluation, just use the following commands:
```
# ===== zero-shot setting =====
# "1" is the GPU id, "movie" or "book" is the target domain, "netflix" or "amazon-book" is the source domain
bash ./shell/zero_shot.sh 1 movie netflix
bash ./shell/zero_shot.sh 1 book amazon-book

# ===== few-shot setting =====
# "1" is the GPU id, "0" is the random seed, "movie" or "book" is the target domain
bash ./shell/few_shot.sh 1 0 movie
bash ./shell/few_shot.sh 1 0 book
```
NOTE: Please modify the root_path in shell files to the folder where you store the code.

## 4. Acknowledgements
Our code is based on the implementation of [TALLRec](https://github.com/SAI990323/TALLRec) and [NIR](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec). 
Thanks for their awesome work.