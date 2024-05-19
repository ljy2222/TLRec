import json
import random
import argparse


# 将三阶段数据转化为instrument格式
def three_steps_to_instrument(input_path, output_path, data_name):
    with open(input_path) as f:
        raw_data = json.load(f)
    print(len(raw_data))
    data_list = []
    for data in raw_data:
        if data["Hit"] == 1:
            data_list.append(data)
    print(len(data_list))

    if 'book' in data_name:
        item_name = 'book'
    else:
        item_name = 'movie'
    json_list_step1 = []
    json_list_step2 = []
    json_list_step3 = []
    for data in data_list:
        # step1
        json_list_step1.append({
            "instruction": f"Given the candidate set and the user's watched {item_name}s, summarize the user's preferences briefly.",
            "input": f"{data['Input_1'][:-8]}",  # remove \nAnswer:
            "output": f"{data['Predictions_1']}",
        })

        # step1 + step2
        json_list_step2.append({
            "instruction": f"Given the candidate set, the user's watched {item_name}s and the user's preferences, select the most featured {item_name}s (at most 5 {item_name}s) from the watched {item_name}s according to the user's preferences in descending order (Format: [no. a watched {item_name}]).",
            "input": f"{data['Input_2'][:-8]}",
            "output": f"{data['Predictions_2']}",
        })

        # step1 + step2 + step3
        json_list_step3.append({
            "instruction": f"Given the candidate set, the user's watched {item_name}s, the user's preferences and the most featured {item_name}s, recommend 10 {item_name}s from the candidate set similar to the selected {item_name}s the user has watched (Format: [no. a watched {item_name} - a candidate {item_name}])",
            "input": f"{data['Input_3'][:-8]}",
            "output": f"{data['Predictions']}",
        })

    with open(output_path + 'step1.json', 'w') as f:
        json.dump(json_list_step1, f, indent=4)

    with open(output_path + 'step12.json', 'w') as f:
        json.dump(json_list_step2, f, indent=4)

    with open(output_path + 'step123.json', 'w') as f:
        json.dump(json_list_step3, f, indent=4)


# nohup python -u three_stage_0_NIR.py --random_seed 2023 > ./nohup_logs/nohup_len_8_numcand_19_seed_2023.txt 2>&1 &
# nohup python -u three_stage_0_NIR.py --random_seed 0 > ./nohup_logs/nohup_len_8_numcand_19_seed_0.txt 2>&1 &
# nohup python -u three_stage_0_NIR.py --random_seed 1 > ./nohup_logs/nohup_len_8_numcand_19_seed_1.txt 2>&1 &
# nohup python -u three_stage_0_NIR.py --random_seed 2 > ./nohup_logs/nohup_len_8_numcand_19_seed_2.txt 2>&1 &
if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--length_limit', type=int, default=8, help='')
    parser.add_argument('--num_cand', type=int, default=19, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    parser.add_argument('--api_key', type=str, default="sk-", help="")
    args = parser.parse_args()

    # 参数设置
    length_limit = args.length_limit  # 用户历史记录的长度
    total_i = args.num_cand  # 候选item的数量
    num_u = 12  # 计算候选集合的时候参考几个用户
    rseed = args.random_seed
    random.seed(rseed)  # 随机种子

    data_name = 'amazon-book'  # ml-100k, ml-1m, amazon-book
    data_root = f'./three_steps_data/{data_name}/'  # 原始数据的地址
    # input_path = f"{data_root}results_multi_prompting_len_8_numcand_19_seed_2023.json"
    input_path = f"{data_root}three_steps.json"
    output_path = f"{data_root}"
    three_steps_to_instrument(input_path, output_path, data_name)
