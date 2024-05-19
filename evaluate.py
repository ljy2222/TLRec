import os
import sys
import json
import fire
import torch
from tqdm import tqdm
torch.set_num_threads(1)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
device = "cuda"


def evaluate(
    instructions,
    inputs=None,
    temperature=0,
    top_p=1.0,
    top_k=40,
    num_beams=1,
    max_new_tokens=128,
    tokenizer=None,
    model=None,
    **kwargs,
):
    prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens
        )
    scores = generation_output.scores[0].softmax(dim=-1)
    logits = torch.tensor(scores[:, [8241, 3782]], dtype=torch.float32).softmax(dim=-1)
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    output = [_.split('Response:\n')[-1] for _ in output]

    return output, logits.tolist()


def evaluate_main(model, tokenizer, test_data):
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # evaluate
    instructions = [_['instruction'] for _ in test_data]
    inputs = [_['input'] for _ in test_data]

    outputs = []
    logits = []
    pred = []
    def batch(list, batch_size=16):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
        instructions, inputs = batch
        output, logit = evaluate(instructions, inputs, tokenizer=tokenizer, model=model)
        outputs = outputs + output
        logits = logits + logit
    for i, test in tqdm(enumerate(test_data)):
        test_data[i]['predict'] = outputs[i]
        test_data[i]['logits'] = logits[i]
        pred.append(logits[i][0])

    return pred


def main(args):
    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"

    # evaluate different settings
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)
        gold = [int(_['output'] == 'Yes.') for _ in test_data]

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # lora weights
    print(f'==========lora weights: {args.lora_weights} start==========')
    if not args.lora_weights:  # base model
        pass
    else:
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    pred = evaluate_main(model, tokenizer, test_data)
    res = roc_auc_score(gold, pred)
    print(f'auc: {res}')
    print(f'==========lora weights: {args.lora_weights} end==========')


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--base_model', type=str, default='/home/ljy/llm_models/llama-7b/')
    parser.add_argument('--lora_weights', type=str, default='')
    parser.add_argument('--test_data_path', type=str, default='')
    args = parser.parse_args()

    main(args)