import csv
import json
import numpy as np
import pandas as pd

data_name = 'amazon-book'  # amazon-book, netflix
data_root = f'/home_nfs/haitao/data/ljy/rec_data/{data_name}/'  # 原始数据的地址
processed_data_root = f'/home_nfs/haitao/data/ljy/CoTRec/instrument_data/{data_name}/'  # 处理后数据的地址
dataset_size = -10000  # 取最后多少个数据组成数据集
train_dataset_ratio = 0.8
test_dataset_ratio = 0.1
valid_dataset_ratio = 0.1
print(f'preprocess dataset {data_name} from {data_root} and store in {processed_data_root}')

rates = pd.read_csv(data_root + '/Books_rating.csv')[['Id', 'Title', 'User_id', 'review/score', 'review/time']]
rates = rates.drop_duplicates(subset=['Title', 'User_id'])
movies = pd.read_csv(data_root + '/books_data.csv')[['Title', 'authors', 'publishedDate']]
movies = movies.drop_duplicates(subset='Title')
data = pd.merge(rates, movies, on='Title', how='inner')
data['publishedDate'] = data['publishedDate'].str.extract(r'(\d{4})')
data['review/score'] = data['review/score'].astype(int)
data = data.dropna()
data = data.astype(str)
data = data.drop_duplicates(subset=['Title', 'User_id'], keep='first')

movies = data.loc[:, ['Id', 'Title', 'authors', 'publishedDate']].drop_duplicates(subset='Id')
users = data.loc[:, ['User_id']].drop_duplicates(subset='User_id')
movie_names = movies['Title'].values.tolist()  # movie_names[0] = 'Toy Story (1995)'，电影名字
movie_authors = movies['authors'].values.tolist()
user_ids = users['User_id'].values.tolist()  # user_ids[0] = '1'，用户id
movie_ids = movies['Id'].values.tolist()  # movie_ids[0] = '1'，电影id

interaction_dicts = dict()
for line in data.values:
    user_id, movie_id, rating, timestamp = line[2], line[0], line[3], line[4]  # 用户id，电影id，分数，时间戳
    if user_id not in interaction_dicts:  # 用户首次出现，记录每个用户看过的电影以及对应的分数和时间戳
        interaction_dicts[user_id] = {
            'movie_id': [],
            'rating': [],
            'timestamp': [],
        }
    interaction_dicts[user_id]['movie_id'].append(movie_id)
    interaction_dicts[user_id]['rating'].append(int(int(rating) > 4))  # 大于3分表示喜欢，小于3分表示不喜欢
    interaction_dicts[user_id]['timestamp'].append(timestamp)

movie_ids_2_names = dict()
with open(processed_data_root + 'item_mapping.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'movie_name'])
    for movie_id, movie_name, movie_author in zip(movie_ids, movie_names, movie_authors):
        writer.writerow([movie_id, movie_name, movie_author])
        movie_ids_2_names[movie_id] = (movie_name, movie_author[2:-2])

sequential_interaction_list = []
seq_len = 10
for user_id in interaction_dicts:
    temp = zip(interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'],
               interaction_dicts[user_id]['timestamp'])
    temp = sorted(temp, key=lambda x: x[2])  # 按照时间戳进行排序
    result = zip(*temp)
    interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id][
        'timestamp'] = [list(_) for _ in result]
    for i in range(10, len(interaction_dicts[user_id]['movie_id'])):  # user id，历史movie id，历史打分，当前movie id，当前打分，时间戳
        sequential_interaction_list.append(
            [user_id, interaction_dicts[user_id]['movie_id'][i - seq_len:i],
             interaction_dicts[user_id]['rating'][i - seq_len:i], interaction_dicts[user_id]['movie_id'][i],
             interaction_dicts[user_id]['rating'][i], interaction_dicts[user_id]['timestamp'][i].strip('\n')]
        )

sequential_interaction_list = sequential_interaction_list[dataset_size:]  # 10000 records

import csv

# save the csv file for baselines
with open(processed_data_root + '/train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[:int(len(sequential_interaction_list) * train_dataset_ratio)])
with open(processed_data_root + '/valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[
                     int(len(sequential_interaction_list) * train_dataset_ratio):int(len(sequential_interaction_list) * (train_dataset_ratio + valid_dataset_ratio))])
with open(processed_data_root + '/test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list) * (train_dataset_ratio + valid_dataset_ratio)):])

def csv_to_json(input_path, output_path):
    data = pd.read_csv(input_path)
    json_list = []
    for index, row in data.iterrows():
        row['history_movie_id'] = eval(row['history_movie_id'])
        row['history_rating'] = eval(row['history_rating'])
        L = len(row['history_movie_id'])
        preference = []
        unpreference = []
        for i in range(L):
            if int(row['history_rating'][i]) == 1:
                # preference.append(movie_names[int(row['history_movie_id'][i]) - 1])
                tmp_movie_name, tmp_movie_author = movie_ids_2_names[str(row['history_movie_id'][i])]
                preference.append("\"" + tmp_movie_name + "\"" + " written by " + tmp_movie_author)
            else:
                # unpreference.append(movie_names[int(row['history_movie_id'][i]) - 1])
                tmp_movie_name, tmp_movie_author = movie_ids_2_names[str(row['history_movie_id'][i])]
                unpreference.append("\"" + tmp_movie_name + "\"" + " written by " + tmp_movie_author)
        # target_movie = movie_names[int(row['movie_id']) - 1]
        target_movie_name, target_movie_author = movie_ids_2_names[str(row['movie_id'])]
        preference_str = ""
        unpreference_str = ""
        for i in range(len(preference)):
            if i == 0:
                preference_str += preference[i]
            else:
                preference_str += ", " + preference[i]
        for i in range(len(unpreference)):
            if i == 0:
                unpreference_str += unpreference[i]
            else:
                unpreference_str += ", " + unpreference[i]
        target_movie_str = "\"" + target_movie_name + "\"" + " written by " + target_movie_author
        target_preference = int(row['rating'])
        target_preference_str = "Yes." if target_preference == 1 else "No."
        json_list.append({
            "instruction": "Given the user's preference and unpreference, identify whether the user will like the target book by answering \"Yes.\" or \"No.\".",
            "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nWhether the user will like the target book {target_movie_str}?",
            "output": target_preference_str,
        })

    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)


# generate the json file for the TALLRec
csv_to_json(processed_data_root + '/train.csv', processed_data_root + '/train.json')
csv_to_json(processed_data_root + '/valid.csv', processed_data_root + '/valid.json')
csv_to_json(processed_data_root + '/test.csv', processed_data_root + '/test.json')