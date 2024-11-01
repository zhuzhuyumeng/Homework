from surprise import Dataset, Reader
from surprise import AlgoBase, PredictionImpossible

class MyOwnAlgorithm(AlgoBase):
    def __init__(self, sim_options={}, bsl_options={}):
        AlgoBase.__init__(self, sim_options=sim_options, bsl_options=bsl_options)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.bu, self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")

        neighbors = [(v, self.sim[u, v]) for (v, r) in self.trainset.ir[i]]
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        print("The 3 nearest neighbors of user", str(u), "are:")
        for v, sim_uv in neighbors[:3]:
            print(f"user {v} with sim {sim_uv:1.2f}")

        bsl = self.trainset.global_mean + self.bu[u] + self.bi[i]
        return bsl

# Load the dataset
data = Dataset.load_builtin('ml-100k')
reader = Reader(line_format='user item rating timestamp', sep='\t')
trainset = data.build_full_trainset()
# Instantiate and fit the algorithm
algo = MyOwnAlgorithm()
algo.fit(trainset)

# Make a prediction
user_id = 196
item_id = 302
prediction = algo.estimate(user_id, item_id)
print(f'Estimated rating for user {user_id} on item {item_id}: {prediction}')
