import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split

# 获取数据集
train_data = pd.read_table('../ml-100k.train.rating',sep='\t',names=['UserId', 'ItemId', 'ratings', 'timestamps'])
test_data = pd.read_table('../ml-100k.test.rating',sep='\t',names=['UserId', 'ItemId', 'ratings', 'timestamps'])
# 转化
reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(train_data[['UserId', 'ItemId', 'ratings']],reader)
testset = Dataset.load_from_df(test_data[['UserId', 'ItemId', 'ratings']],reader)
trainset = trainset.build_full_trainset()
testset = test_data[['UserId','ItemId','ratings']].values

# 建立模型
algo = SVD(biased=False,n_epochs=20)
algo2 = SVD(biased=False,n_epochs=20,lr_all=0.003)
# biased=True的效果要好点
# 这个数据集n_factors=少一点效果好
# 迭代次数高误差变高了啊,大差不差吧
algo.fit(trainset)
algo2.fit(trainset)
predictions1 = algo.test(testset)
predictions2 = algo2.test(testset)

rmse1 = accuracy.rmse(predictions1)
rmse2 = accuracy.rmse(predictions2)
