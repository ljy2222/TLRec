import re
import time
import json
import random
import argparse
import numpy as np
from openai import OpenAI


# user过滤
def sort_uf_items(target_seq, us, num_u, num_i):
    candidate_movies_dict = {}
    sorted_us = sorted(list(enumerate(us)), key=lambda x: x[-1], reverse=True)[:num_u]
    dvd = sum([e[-1] for e in sorted_us])
    for us_i, us_v in sorted_us:
        us_w = us_v * 1.0 / dvd
        us_elem = data_ml_100k[us_i]
        us_seq_list = us_elem[0].split(' | ')
        for us_m in us_seq_list:
            if us_m not in target_seq:
                if us_m not in candidate_movies_dict:
                    candidate_movies_dict[us_m] = 0.
                candidate_movies_dict[us_m] += us_w
    candidate_pairs = list(sorted(candidate_movies_dict.items(), key=lambda x: x[-1], reverse=True))
    candidate_items = [e[0] for e in candidate_pairs][:num_i]
    return candidate_items


# 传入open ai接口，得到对应的response
def get_response(client, input):
    response = ''
    try_nums = 5
    while try_nums:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # gpt-3.5-turbo
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input},
                ],
                max_tokens=512,  # 输出结果的最大token数
                temperature=0.2,  # 控制生成文本的多样性，强调创造性
                top_p=1,  # 控制生成文本的多样性，强调可控性
                frequency_penalty=0,  # 惩罚训练数据中出现频率较高的词汇
                presence_penalty=0,  # 惩罚生成文本中频繁出现的词汇
                n=1  # 候选文本数
            )
            try_nums = 0
        except Exception as e:
            if 'exceeded your current quota' in str(e):
                print(e)
            time.sleep(1)
            try_nums -= 1
    return response

# nohup python -u three_stage_0_NIR.py --data_name amazon-book > ./nohup_logs/nohup_amazon_book.txt 2>&1 &
# cat ./nohup_logs/nohup_amazon_book.txt
# nohup python -u three_stage_0_NIR.py --data_name netflix > ./nohup_logs/nohup_netflix.txt 2>&1 &
# cat ./nohup_logs/nohup_netflix.txt
if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--length_limit', type=int, default=8, help='')
    parser.add_argument('--num_cand', type=int, default=19, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    parser.add_argument('--api_key', type=str, default="sk-", help="")
    parser.add_argument('--data_name', type=str, default="amazon-book", help="")
    args = parser.parse_args()

    # 参数设置
    data_name = args.data_name  # ml-100k, ml-1m, amazon-book
    length_limit = args.length_limit  # 用户历史记录的长度
    total_i = args.num_cand  # 候选item的数量
    num_u = 12  # 计算候选集合的时候参考几个用户
    rseed = args.random_seed
    random.seed(rseed)  # 随机种子

    data_root = f'/home_nfs/haitao/data/ljy/TALLRec/processed_data/{data_name}/'  # 原始数据的地址
    with open(f"{data_root}train.json") as f:
        raw_data = json.load(f)[:3000]
    data_ml_100k = []
    for data in raw_data:
        if data['output'] == 'Yes.':
            movies = re.findall(r'"(.*?)"', data['input'])
            data_ml_100k.append([" | ".join(movies[:-1]), movies[-1]])
    print(len(data_ml_100k))

    # 自己买的
    api_key = "sk-6pgb149873665d346a550a6db87545814ba3eee81bcMq76o"
    api_base = "https://api.gptsapi.net/v1"
    # # 蚂蚁1
    # api_key = "sk-ZVWyoaNXLodnfsqbD165945fAcA44c4fA5265188D3Fb48C9"
    # api_base = "https://pro.aiskt.com/v1"
    # # 蚂蚁2
    # api_key = "sk-aWQspIYghLrbFGS607D6F3E66dBe47EaA7DeDfA13a620eF9"
    # api_base = "https://one.aiskt.com/v1"
    client = OpenAI(api_key=api_key, base_url=api_base)

    # user-item
    u_item_dict = {}  # 每个item的编号
    u_item_p = 0
    for elem in data_ml_100k:
        seq_list = elem[0].split(' | ')
        for movie in seq_list:
            if movie not in u_item_dict:
                u_item_dict[movie] = u_item_p
                u_item_p += 1
    u_item_len = len(u_item_dict)
    print('u_item_len:', u_item_len)

    user_list = []
    for i, elem in enumerate(data_ml_100k):
        item_hot_list = [0 for ii in range(u_item_len)]
        seq_list = elem[0].split(' | ')
        for movie in seq_list:
            item_pos = u_item_dict[movie]
            item_hot_list[item_pos] = 1
        user_list.append(item_hot_list)
    user_matrix = np.array(user_list)
    print('user_matrix.shape:', user_matrix.shape)
    user_matrix_sim = np.dot(user_matrix, user_matrix.transpose())  # 用户的相似度矩阵

    # user筛选
    count = 0
    total = 0
    cand_ids = []
    id_list = list(range(0, len(data_ml_100k)))
    for i in id_list:
        elem = data_ml_100k[i]
        seq_list = elem[0].split(' | ')
        candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i)
        if elem[-1] in candidate_items:
            count += 1
            cand_ids.append(i)
        total += 1
    print(f'count/total: {count}/{total} = {count * 1.0 / total}')
    print('-----------------\n')

    # our version
    if 'book' in data_name:
        item_name = 'book'
        input = "Candidate Set (candidate books): {}\nThe books that the user has watched (watched books): {}\n"
    else:
        item_name = 'movie'
        input = "Candidate Set (candidate movies): {}\nThe movies that the user has watched (watched movies): {}\n"
    instruction_1 = f"Step 1: What features are most important to the user when selecting {item_name}s (Summarize the user's preferences briefly)?"
    instruction_2 = f"Step 2: Selecting the most featured {item_name}s (at most 5 {item_name}s) from the watched {item_name}s according to the user's preferences in descending order (Format: [no. a watched {item_name}])."
    instruction_3 = f"Step 3: Can you recommend 10 {item_name}s from the Candidate Set similar to the selected {item_name}s the user has watched (Format: [no. a watched {item_name} - a candidate {item_name}])?"

    temp_1 = input + instruction_1 + '\nAnswer:'
    temp_2 = input + instruction_1 + '\nAnswer: {}\n' + instruction_2 + '\nAnswer:'
    temp_3 = input + instruction_1 + '\nAnswer: {}\n' + instruction_2 + '\nAnswer: {}\n' + instruction_3 + '\nAnswer:'

    # # 原论文中的template
    # temp_1 = """
    # Candidate Set (candidate movies): {}.
    # The movies I have watched (watched movies): {}.
    # Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)?
    # Answer:
    # """
    #
    # temp_2 = """
    # Candidate Set (candidate movies): {}.
    # The movies I have watched (watched movies): {}.
    # Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)?
    # Answer: {}.
    # Step 2: Selecting the most featured movies from the watched movies according to my preferences (Format: [no. a watched movie.]).
    # Answer:
    # """
    #
    # temp_3 = """
    # Candidate Set (candidate movies): {}.
    # The movies I have watched (watched movies): {}.
    # Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)?
    # Answer: {}.
    # Step 2: Selecting the most featured movies (at most 5 movies) from the watched movies according to my preferences in descending order (Format: [no. a watched movie.]).
    # Answer: {}.
    # Step 3: Can you recommend 10 movies from the Candidate Set similar to the selected movies I've watched (Format: [no. a watched movie - a candidate movie])?.
    # Answer:
    # """

    count = 0
    total = 0
    results_data = []
    for i in cand_ids:
        print(len(cand_ids))
        elem = data_ml_100k[i]
        seq_list = elem[0].split(' | ')[::-1]
        candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i)
        random.shuffle(candidate_items)

        # step1
        input_1 = temp_1.format(', '.join(f'"{item}"' for item in candidate_items),
                                ', '.join(f'"{item}"' for item in seq_list[-length_limit:]))
        response = get_response(client, input_1)
        if response == '':
            continue
        predictions_1 = response.choices[0].message.content

        # step2
        input_2 = temp_2.format(', '.join(f'"{item}"' for item in candidate_items),
                                ', '.join(f'"{item}"' for item in seq_list[-length_limit:]), predictions_1)
        response = get_response(client, input_2)
        if response == '':
            continue
        predictions_2 = response.choices[0].message.content

        # step3
        input_3 = temp_3.format(', '.join(f'"{item}"' for item in candidate_items),
                                ', '.join(f'"{item}"' for item in seq_list[-length_limit:]), predictions_1,
                                predictions_2)
        response = get_response(client, input_3)
        if response == '':
            continue
        predictions = response.choices[0].message.content

        if predictions is None:
            continue

        # 是否命中
        hit_ = 0
        if elem[1] in predictions:
            count += 1
            hit_ = 1
        total += 1

        # print (f"input_1: \n{input_1}")
        # print (f"predictions_1: {predictions_1}\n")
        # print (f"input_2: \n{input_2}")
        # print (f"predictions_2: {predictions_2}\n")
        print(f"input_3: \n{input_3}")
        print(f"predictions: \n{predictions}")
        print(f"GT: {elem[1]}")
        print(f'PID: {i}; count/total: {count}/{total} = {count * 1.0 / total}\n')

        result_json = {"PID": i,  # 样本序号
                       "Input_1": input_1,  # 第一阶段的输入
                       "Input_2": input_2,  # 第二阶段的输入
                       "Input_3": input_3,  # 第三阶段的输入
                       "Predictions_1": predictions_1,  # 第一阶段的输出
                       "Predictions_2": predictions_2,  # 第二阶段的输出
                       "Predictions": predictions,  # 第三阶段的输出
                       "GT": elem[1],  # GT
                       'Hit': hit_,  # 是否命中
                       'Count': count,  # 命中的次数
                       'Current_total': total,  # 总次数
                       'Hit@10': count * 1.0 / total}  # 命中率
        results_data.append(result_json)

    file_dir = f"./three_steps_data/{data_name}/results_multi_prompting_len_{length_limit}_numcand_{total_i}_seed_{rseed}.json"
    with open(file_dir, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)