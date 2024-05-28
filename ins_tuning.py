import os
import sys
import torch
import transformers
from functools import partial
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import EarlyStoppingCallback
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from utils import *


def train(args):
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    train_data = load_dataset("json", data_files=args.train_data_path)
    val_data = load_dataset("json", data_files=args.val_data_path)
    train_data = (train_data["train"].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer)))
    val_data = (val_data["train"].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer)))

    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    model.print_trainable_parameters()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=int(args.batch_size) // int(args.micro_batch_size),
            warmup_steps=20,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=10,
            save_steps=10,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            greater_is_better=True,
            ddp_find_unused_parameters=None,
            group_by_length=False
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=cal_metric,
        preprocess_logits_for_metrics=preprocess_logits_for_metric,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--train_data_path", type=str, default="")
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ': ' + str(args.__dict__[k]))

    train(args)