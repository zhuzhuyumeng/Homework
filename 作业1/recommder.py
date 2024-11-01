import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy
# train_data = pd.read_table('ml-100k.train.rating',sep='\t',names=['UserId', 'ItemId', 'ratings', 'timestamps'])
# test_data = pd.read_table('ml-100k.test.rating',sep='\t',names=['UserId', 'ItemId', 'ratings', 'timestamps'])

data = Dataset.load_builtin('ml-100k')
trainset,testset = train_test_split(data,test_size=0.25)
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)

accuracy.rmse(predictions)
cross_validate(algo,data,measures=['RMSE','MAE'],cv=5,verbose=True)
