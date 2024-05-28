import sys
import json
import torch
from tqdm import tqdm
from peft import PeftModel
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils import *


def get_output(instructions, inputs, tokenizer, model):
    prompt = [generate_prompt_eval(instruction, input) for instruction, input in zip(instructions, inputs)]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    generation_config = GenerationConfig(
        temperature=0,
        top_p=1,
        top_k=40,
        num_beams=1
    )
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128
        )
    scores = generation_output.scores[0].softmax(dim=-1)
    logits = torch.tensor(scores[:, [8241, 3782]], dtype=torch.float32).softmax(dim=-1)
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    output = [_.split("Response:\n")[-1] for _ in output]
    return output, logits.tolist()


def predict(model, tokenizer, test_data):
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.half()
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    instructions = [_["instruction"] for _ in test_data]
    inputs = [_["input"] for _ in test_data]
    outputs = []
    logits = []
    pred = []
    for i, batch in tqdm(enumerate(zip(get_batch(instructions, batch_size=16), get_batch(inputs, batch_size=16)))):
        instructions, inputs = batch
        output, logit = get_output(instructions, inputs, tokenizer=tokenizer, model=model)
        outputs = outputs + output
        logits = logits + logit
    for i, test in tqdm(enumerate(test_data)):
        test_data[i]["predict"] = outputs[i]
        test_data[i]["logits"] = logits[i]
        pred.append(logits[i][0])
    return pred


def evaluate(args):
    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)
        gold = [int(_["output"] == "Yes.") for _ in test_data]
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.lora_weights:
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0}
        )
    pred = predict(model, tokenizer, test_data)
    auc = roc_auc_score(gold, pred)
    print(f"===== auc: {auc} =====")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--test_data_path", type=str, default="")
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ': ' + str(args.__dict__[k]))

    evaluate(args)