import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import PredictionImpossible
from surprise import PredictionImpossible

def MyOwnAlgorithm(AlgoBase):
    def __init__(self,sim_options={},bsl_options={}):
        AlgoBase.__init__(self,sim_options=sim_options,bsl_options=bsl_options)

    def fit(self,trainset):
        AlgoBase.fit(self,trainset)
        self.bu,self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()
        return self

    def estimate(self,u,i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        neighbors = [(v,self.sim[u,v]) for (v,r) in self.trainset.ir[i]]
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
        print("The 3 nearest neighbors of user", str(u), "are:")
        for v,sim_uv in neighbors[:3]:
            print(f'user {v} with sim {sim_uv:1.2f}')
        bsl = self.trainset.global_mean + self.bu[u] + self.bi[i]
        return bsl


train_data = pd.read_table('../ml-100k.train.rating', sep='\t', names=['UserId', 'ItemId', 'ratings', 'timestamps'])
test_data = pd.read_table('../ml-100k.test.rating', sep='\t', names=['UserId', 'ItemId', 'ratings', 'timestamps'])
# 转化
reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(train_data[['UserId', 'ItemId', 'ratings']],reader)
testset = Dataset.load_from_df(test_data[['UserId', 'ItemId', 'ratings']],reader)
#
trainset = trainset.build_full_trainset()
# 建立模型

algo = MyOwnAlgorithm()
algo.fit(testset)


# print(f'SVD的RMSE:{rmse}')