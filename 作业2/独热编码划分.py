import pandas as pd
import re
from io import StringIO  # Import StringIO from the io module

def load_file_with_mixed_delimiters(file_path,names=None):
    # 读取文件
    with open(file_path,'r', encoding='ISO-8859-1') as file:
        content = file.read()
    # 使用正则表达式替换分隔符
    # 将 "||" 替换为 "|"，然后将 "|" 和 "||" 统一为单个分隔符
    cleaned_content = re.sub(r'\|\|+', '|', content)
    # 使用分隔符 "|" 加载 DataFrame
    return pd.read_csv(StringIO(cleaned_content), sep='|',header=None,names=names)

# 2. 加载类别文件
genres_file_path = 'F:/data/ml-100k/u.genre'  # 替换为实际文件路径
genres_column_names = ['genres_name', 'genres_id']
genres_df = load_file_with_mixed_delimiters(genres_file_path,genres_column_names)
genres_df.to_csv('F:\\1120241382\\genre.csv',mode='w')
genres_list = []
for i in range(len(genres_df)):
    genres_list.append(genres_df.iloc[i][0])


# 1. 加载电影文件
movies_file_path = 'F:/data/ml-100k/u.item'  # 替换为实际文件路径
movies_column_names = ['item_id', 'title', 'release_date', 'url']+genres_list
movies_df = load_file_with_mixed_delimiters(movies_file_path,movies_column_names)
movies_df.to_csv('F:\\1120241382\\onehot.csv',mode='w')

genre_columns = movies_df.columns[5:]  # 从第三列开始都是种类
movies_df['genres'] = movies_df[genre_columns].apply(lambda x: [genre for genre, val in zip(genre_columns, x) if val == 1], axis=1)


# 5. 显示合并后的数据
print("合并后的电影与种类数据:")
print(movies_df.head())

# 7. 提取需要的列
final_df = movies_df[['title', 'genres']]

# 8. 显示最终结果
print("电影与对应的种类:")
print(final_df.head())
# movies_df.to_csv('F:\\1120241382\\genres.csv',mode='w')
