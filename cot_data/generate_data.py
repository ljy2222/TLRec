import re
import time
import json
import random
import argparse
import numpy as np
from openai import OpenAI


# user filtering
def sort_uf_items(target_seq, us, user_num, candidate_num):
    candidate_movies_dict = {}
    sorted_us = sorted(list(enumerate(us)), key=lambda x: x[-1], reverse=True)[:user_num]
    dvd = sum([e[-1] for e in sorted_us])
    for us_i, us_v in sorted_us:
        us_w = us_v * 1.0 / dvd
        us_elem = processed_data[us_i]
        us_seq_list = us_elem[0].split(" | ")
        for us_m in us_seq_list:
            if us_m not in target_seq:
                if us_m not in candidate_movies_dict:
                    candidate_movies_dict[us_m] = 0.
                candidate_movies_dict[us_m] += us_w
    candidate_pairs = list(sorted(candidate_movies_dict.items(), key=lambda x: x[-1], reverse=True))
    candidate_items = [e[0] for e in candidate_pairs][:candidate_num]
    return candidate_items


# get responses from OpenAI’s GPT-4 API
def get_response(client, input):
    response = ''
    try_nums = 5
    while try_nums:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": input},
                ],
                max_tokens=512,
                temperature=0.2,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=1
            )
            try_nums = 0
        except Exception as e:
            print(e)
            time.sleep(1)
            try_nums -= 1
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/your/root/path/TLRec/")
    parser.add_argument("--input_path", type=str, default="/instruction_data/")
    parser.add_argument("--data_name", type=str, default="netflix")
    parser.add_argument("--output_path", type=str, default="/cot_data/")
    args = parser.parse_args()

    sequence_length, candidate_num, user_num, seed = 10, 19, 12, 0
    random.seed(seed)
    api_key = "sk-xxx"
    api_base = "https://xxx"
    client = OpenAI(api_key=api_key, base_url=api_base)

    with open(f"{args.root_path}{args.input_path}{args.data_name}/train.json") as f:
        raw_data = json.load(f)
    processed_data = []
    for sample in raw_data:
        if sample["output"] == "Yes.":
            items = re.findall(r'"(.*?)"', sample["input"])
            processed_data.append([" | ".join(items[:-1]), items[-1]])

    user_item_dict = {}
    user_item_p = 0
    for sample in processed_data:
        seq_list = sample[0].split(" | ")
        for item in seq_list:
            if item not in user_item_dict:
                user_item_dict[item] = user_item_p
                user_item_p += 1
    user_item_len = len(user_item_dict)

    user_list = []
    for i, sample in enumerate(processed_data):
        item_hot_list = [0 for ii in range(user_item_len)]
        seq_list = sample[0].split(" | ")
        for item in seq_list:
            item_pos = user_item_dict[item]
            item_hot_list[item_pos] = 1
        user_list.append(item_hot_list)
    user_matrix = np.array(user_list)
    user_matrix_sim = np.dot(user_matrix, user_matrix.transpose())

    count, total = 0, 0
    candidate_ids = []
    id_list = list(range(0, len(processed_data)))
    for i in id_list:
        sample = processed_data[i]
        seq_list = sample[0].split(" | ")
        candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], user_num=user_num, candidate_num=candidate_num)
        if sample[-1] in candidate_items:
            count += 1
            candidate_ids.append(i)
        total += 1
    print(f"count/total: {count}/{total} = {count * 1.0 / total}\n==========\n")

    if "book" in args.data_name:
        item_name = "book"
        input = "Candidate Set (candidate books): {}\nThe books that the user has watched (watched books): {}\n"
    else:
        item_name = "movie"
        input = "Candidate Set (candidate movies): {}\nThe movies that the user has watched (watched movies): {}\n"
    instruction_1 = f"Step 1: What features are most important to the user when selecting {item_name}s (Summarize the user's preferences briefly)?"
    instruction_2 = f"Step 2: Selecting the most featured {item_name}s (at most 5 {item_name}s) from the watched {item_name}s according to the user's preferences in descending order (Format: [no. a watched {item_name}])."
    instruction_3 = f"Step 3: Can you recommend 10 {item_name}s from the Candidate Set similar to the selected {item_name}s the user has watched (Format: [no. a watched {item_name} - a candidate {item_name}])?"
    temp_1 = input + instruction_1 + "\nAnswer:"
    temp_2 = input + instruction_1 + "\nAnswer: {}\n" + instruction_2 + "\nAnswer:"
    temp_3 = input + instruction_1 + "\nAnswer: {}\n" + instruction_2 + "\nAnswer: {}\n" + instruction_3 + "\nAnswer:"

    count, total = 0, 0
    result_data = []
    for i in candidate_ids:
        sample = processed_data[i]
        seq_list = sample[0].split(" | ")[::-1]
        candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], user_num=user_num, candidate_num=candidate_num)
        random.shuffle(candidate_items)
        # stage 1
        input_1 = temp_1.format(", ".join(f'"{item}"' for item in candidate_items),
                                ", ".join(f'"{item}"' for item in seq_list[-sequence_length:]))
        response = get_response(client, input_1)
        if response == "":
            continue
        predictions_1 = response.choices[0].message.content
        # stage 2
        input_2 = temp_2.format(", ".join(f'"{item}"' for item in candidate_items),
                                ", ".join(f'"{item}"' for item in seq_list[-sequence_length:]), predictions_1)
        response = get_response(client, input_2)
        if response == "":
            continue
        predictions_2 = response.choices[0].message.content
        # stage 3
        input_3 = temp_3.format(", ".join(f'"{item}"' for item in candidate_items),
                                ", ".join(f'"{item}"' for item in seq_list[-sequence_length:]), predictions_1,
                                predictions_2)
        response = get_response(client, input_3)
        if response == "":
            continue
        predictions = response.choices[0].message.content
        if predictions is None:
            continue
        # hit or not
        hit = 0
        if sample[1] in predictions:
            count += 1
            hit = 1
        total += 1
        print(f"input_3: \n{input_3}")
        print(f"predictions: \n{predictions}")
        print(f"GT: {sample[1]}")
        print(f"PID: {i}; count/total: {count}/{total} = {count * 1.0 / total}\n")

        result_json = {"PID": i,
                       "Input_1": input_1,
                       "Input_2": input_2,
                       "Input_3": input_3,
                       "Predictions_1": predictions_1,
                       "Predictions_2": predictions_2,
                       "Predictions": predictions,
                       "GT": sample[1],
                       "Hit": hit,
                       "Count": count,
                       "Current_total": total,
                       "Hit@10": count * 1.0 / total}
        result_data.append(result_json)

    with open(f"{args.root_path}{args.output_path}{args.data_name}/three_stages.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)