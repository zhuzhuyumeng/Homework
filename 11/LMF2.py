import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 1. 加载数据集
# 加载评分数据
ratings = pd.read_csv('F:/data/ml-1m/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'],
                      engine='python')

# 加载电影数据
movies = pd.read_csv('F:/data/ml-1m/movies.dat', sep='::', encoding='latin-1', header=None, engine='python',
                     names=['item_id', 'title', 'genres'])

# 处理电影的类别，将每个电影的类别按'|'分隔
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# 使用MultiLabelBinarizer自动编码所有类别
mlb = MultiLabelBinarizer()
movies = movies.join(pd.DataFrame(mlb.fit_transform(movies.pop('genres')),
                                  columns=mlb.classes_,
                                  index=movies.index))

# 选择需要推荐的电影类别
categories = ['Mystery', 'War', 'Thriller', 'Romance']

# 2. 构建用户-物品评分矩阵
rating_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# 3. 使用SVD进行矩阵分解
svd = TruncatedSVD(n_components=20)  # 选择20个隐因子
item_factors = svd.fit_transform(rating_matrix.T)

# 创建一个DataFrame来存储每个物品的隐向量
item_factors_df = pd.DataFrame(item_factors, index=rating_matrix.columns)
print(item_factors_df)

# 4. 定义函数：计算每个类别中权重最高的物品
def get_top_items_for_category(category, n_top=10):
    # 筛选属于该类别的物品
    category_items = movies[movies[category] == 1]

    # 确保有 'item_id' 列
    if 'item_id' not in category_items.columns:
        raise KeyError("'item_id' 列不存在于 movies 数据集中")

    category_item_ids = category_items['item_id'].values

    # 确保这些物品的ID存在于item_factors_df中
    valid_item_ids = [item_id for item_id in category_item_ids if item_id in item_factors_df.index]

    if not valid_item_ids:
        return []  # 如果没有有效的物品ID，返回空列表

    # 提取这些物品的隐向量
    category_item_factors = item_factors_df.loc[valid_item_ids]

    # 计算每个物品在隐类中的权重（用隐向量的和来作为权重）
    category_item_factors['weight'] = category_item_factors.apply(lambda row: row.sum(), axis=1)

    # 按权重排序，选择Top N
    top_items = category_item_factors.sort_values(by='weight', ascending=False).head(n_top)

    # 只返回电影的标题
    return movies[movies['item_id'].isin(top_items.index)]['title'].tolist()

    # 确保在 merge 时 movies 中有 'item_id' 列
    return movies[movies['item_id'].isin(top_items.index)][['title', 'item_id']].merge(
        top_items[['weight']], left_on='item_id', right_index=True)


# 5. 获取每个类别的Top 10电影名称
top_mystery = get_top_items_for_category('Mystery')
top_war = get_top_items_for_category('War')
top_thriller = get_top_items_for_category('Thriller')
top_romance = get_top_items_for_category('Romance')

# 构建一个DataFrame，将类别作为列，Top 10电影名称作为行
top_movies_df = pd.DataFrame({
    'Top 1-10': range(1, 11),  # Top 10 序号作为第一列
    'Mystery': top_mystery,
    'War': top_war,
    'Thriller': top_thriller,
    'Romance': top_romance
})

# 设置列名称
top_movies_df.set_index('Top 1-10', inplace=True)

# 6.输出表格形式
print("Top 10 Movies by Genre:")
print(top_movies_df)
