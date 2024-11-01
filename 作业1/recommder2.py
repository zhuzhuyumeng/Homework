import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split

# 获取数据集
train_data = pd.read_table('../ml-100k.train.rating', sep='\t', names=['UserId', 'ItemId', 'ratings', 'timestamps'])
test_data = pd.read_table('../ml-100k.test.rating', sep='\t', names=['UserId', 'ItemId', 'ratings', 'timestamps'])
# 转化
reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(train_data[['UserId', 'ItemId', 'ratings']],reader)
testset = Dataset.load_from_df(test_data[['UserId', 'ItemId', 'ratings']],reader)
trainset = trainset.build_full_trainset()
# 建立模型
algo = SVD(biased=True)
algo.fit(trainset)
testset = test_data[['UserId','ItemId','ratings']].values
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)

data = Dataset.load_builtin('ml-100k')
trainset2,testset2 = train_test_split(data,test_size=0.2)
algo.fit(trainset2)
predictions = algo.test(testset2)

rmse2 = accuracy.rmse(predictions)

print(f'时间序列划分的RMSE:{rmse}')
print(f'随机划分的RMSE:{rmse2}')



