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

if data_name == 'amazon-book':
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

    # x = data.groupby('User_id').count()['review/score'] > 20
    # considerable_users = x[x].index
    # filtered_rating = data[data['User_id'].isin(considerable_users)]
    # y = filtered_rating.groupby('Title').count()['review/score'] >= 20
    # famous_books = y[y].index
    # data = filtered_rating[filtered_rating['Title'].isin(famous_books)]

    movies = data.loc[:, ['Id', 'Title', 'authors', 'publishedDate']].drop_duplicates(subset='Id')
    # movies['Title'] = movies['Title'] + ' (' + movies['publishedDate'].astype(str) + ')'  # 为title加上年份，防止重名
    users = data.loc[:, ['User_id']].drop_duplicates(subset='User_id')
    movie_names = movies['Title'].values.tolist()  # movie_names[0] = 'Toy Story (1995)'，电影名字
    movie_authors = movies['authors'].values.tolist()
    user_ids = users['User_id'].values.tolist()  # user_ids[0] = '1'，用户id
    movie_ids = movies['Id'].values.tolist()  # movie_ids[0] = '1'，电影id
elif data_name == 'netflix':
    df1 = pd.read_csv(data_root + '/combined_data_1.txt', header=None, names=['Cust_Id', 'Rating', 'Timestamp'], usecols=[0, 1, 2])
    df1['Rating'] = df1['Rating'].astype(float)
    df = df1
    df.index = np.arange(0, len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    df = df[pd.notnull(df['Rating'])]
    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)
    data = df
    data['Rating'] = data['Rating'].astype(int)
    data = data.dropna()
    data = data.astype(str)
    print(df.iloc[::5000000, :])

    def handle_bad_lines(line):
        fields = [str(field) for field in line]
        movie_id = int(fields[0])
        release_year = int(fields[1])
        combined_title = ''.join(fields[2:]).strip()
        return movie_id, release_year, combined_title
    movies = pd.read_csv(data_root + '/movie_titles.csv', names=['MovieId', 'Year', 'Title'],
                                  encoding='ISO-8859-1', engine='python', on_bad_lines=handle_bad_lines)
    movies['MovieId'] = movies['MovieId'].astype(str)
    movies['Year'] = movies['Year'].astype('Int64')
    movies['Title'] = movies['Title'] + ' (' + movies['Year'].astype(str) + ')'  # 为title加上年份，防止重名
    users = data.loc[:, ['Cust_Id']].drop_duplicates(subset='Cust_Id')

    movie_names = movies['Title'].values.tolist()  # movie_names[0] = 'Toy Story (1995)'，电影名字
    user_ids = users['Cust_Id'].values.tolist()  # user_ids[0] = '1'，用户id
    movie_ids = movies['MovieId'].values.tolist()  # movie_ids[0] = '1'，电影id

interaction_dicts = dict()
if data_name == 'amazon-book':
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
elif data_name == 'netflix':
    for line in data.values:
        user_id, movie_id, rating, timestamp = line[0], line[3], line[1], line[2]  # 用户id，电影id，分数，时间戳
        if user_id not in interaction_dicts:  # 用户首次出现，记录每个用户看过的电影以及对应的分数和时间戳
            interaction_dicts[user_id] = {
                'movie_id': [],
                'rating': [],
                'timestamp': [],
            }
        interaction_dicts[user_id]['movie_id'].append(movie_id)
        interaction_dicts[user_id]['rating'].append(int(int(rating) > 3))  # 大于3分表示喜欢，小于3分表示不喜欢
        interaction_dicts[user_id]['timestamp'].append(timestamp)

movie_ids_2_names = dict()
with open(processed_data_root + 'item_mapping.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'movie_name'])
    for movie_id, movie_name, movie_author in zip(movie_ids, movie_names, movie_authors):
        writer.writerow([movie_id, movie_name, movie_author])
        movie_ids_2_names[movie_id] = (movie_name, movie_author)

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
                preference.append(movie_ids_2_names[str(row['history_movie_id'][i])])
            else:
                # unpreference.append(movie_names[int(row['history_movie_id'][i]) - 1])
                unpreference.append(movie_ids_2_names[str(row['history_movie_id'][i])])
        # target_movie = movie_names[int(row['movie_id']) - 1]
        target_movie = movie_ids_2_names[str(row['movie_id'])]
        preference_str = ""
        unpreference_str = ""
        for i in range(len(preference)):
            if i == 0:
                preference_str += "\"" + preference[i] + "\""
            else:
                preference_str += ", \"" + preference[i] + "\""
        for i in range(len(unpreference)):
            if i == 0:
                unpreference_str += "\"" + unpreference[i] + "\""
            else:
                unpreference_str += ", \"" + unpreference[i] + "\""
        target_preference = int(row['rating'])
        target_movie_str = "\"" + target_movie + "\""
        target_preference_str = "Yes." if target_preference == 1 else "No."
        if 'book' in data_name:
            json_list.append({
                "instruction": "Given the user's preference and unpreference, identify whether the user will like the target book by answering \"Yes.\" or \"No.\".",
                "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nWhether the user will like the target book {target_movie_str}?",
                "output": target_preference_str,
            })
        else:
            json_list.append({
                "instruction": "Given the user's preference and unpreference, identify whether the user will like the target movie by answering \"Yes.\" or \"No.\".",
                "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nWhether the user will like the target movie {target_movie_str}?",
                "output": target_preference_str,
            })

    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)


# generate the json file for the TALLRec
csv_to_json(processed_data_root + '/train.csv', processed_data_root + '/train.json')
csv_to_json(processed_data_root + '/valid.csv', processed_data_root + '/valid.json')
csv_to_json(processed_data_root + '/test.csv', processed_data_root + '/test.json')