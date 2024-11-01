import re

import pandas as pd
from sklearn.utils import shuffle


def process_data():
    "载入数据"
    print('Start processing movielens-100k dataset.')
    print('---User Data---')
    users_title = ['UserId', 'Age', 'Gender', 'JobID', 'ZipCode']
    users = pd.read_table(
        'ml-100k/u.user',
        sep='|',
        header=None,
        names=users_title,
        engine='python',
        encoding='ISO-8859-1'
    )
    # names= 将对应的名称到列上
    # header = None 没有标题行,
    # 只保留列名为这些的列
    users = users.filter(regex='UserId|Gender|Age|JobID|ZipCode')
    # 创建一个字典,将用于性别的字符串F,M映射为0,1
    gender_map = {'F': 0, 'M': 1}
    # 对Gender列中的性别值根据gender_map进行转换
    users['Gender'] = users['Gender'].map(gender_map)
    # set函数用于获取Age列中的唯一值,set自动去重
    # enumerate函数将唯一的年龄值与整数索引关联起来,ii是索引值,val是年龄值
    # 字典推导式,将唯一的年龄值映射到对应的整数索引,如果users['Age']中有年龄25和30,且25在集合中的顺序为第0位,30为第一位,则age_map={25:0,30:0}
    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    # 年龄值都被替换为索引值
    users['Age'] = users['Age'].map(age_map)

    print("---Data---")
    ratings_title = ['UserId', 'ItemId', 'ratings', 'timestamps']
    ratings = pd.read_table(
        'ml-100k/u.data',
        sep='\t',
        header=None,
        names=ratings_title,
        engine='python',
        encoding='ISO-8859-1')
    # 你把时间列过滤掉干嘛
    ratings = ratings.filter(regex='UserId|ItemId|ratings|timestamps')

    users_ratings = users.filter(regex='UserId')

    # print(users_ratings)
    ratings_uers = ratings.filter(regex='UserId|ItemId|ratings|timestamps')

    merged_df = pd.merge(ratings_uers, users_ratings)
    # result_df = merged_df
    result_df = merged_df.sort_values(by=['UserId','timestamps'], ascending=[True,False])
    test_df = result_df.groupby('UserId').first().reset_index()
    # 找到要删除的行
    first_rows_indices = result_df.groupby('UserId').head(1).index
    # 删除这些行
    train_df = result_df.drop(first_rows_indices)

    # 打印或保存结果
    # print(negative_samples_df)
    print(test_df)
    # print(train_df)
    train_df.to_csv(
        r'ml-100k.train.rating', index=False, sep='\t', mode='a', header=False)
    test_df.to_csv(
        r'ml-100k.test.rating', index=False, sep='\t', mode='a', header=False)
    # 你和标准差1啊
    print('Process Done.')
    return ratings, users


ratings, users  = process_data()

print('Done')
