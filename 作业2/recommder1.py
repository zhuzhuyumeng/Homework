import random
import math
import pandas as pd
import numpy as np
class SVD():
    def __init__(self,rating_data,F=5,alpha=0.1,lmbda=0.1,max_iter=100):
        self.F = F
        self.P = dict()
        self.Q = dict()
        self.bu = dict()
        self.bi = dict()
        self.mu = 0.0
        self.alpha = alpha
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.rating_data = rating_data

        cnt = 0
        for user,items in self.rating_data.items():
            self.P[user] = [random.random()/math.sqrt(self.F) for x in range(0,F)]
            self.bu[user] = 0
            # 将计数器 cnt 增加该用户评分的物品数量，len(items) 返回用户评分的物品数量。
            cnt += len(items)

            for item,rating in items.items():
                if item not in self.Q:
                    self.Q[item] = [random.random()/math.sqrt(self.F) for x in range(0,F)]
                    self.bi[item] = 0
                self.mu += rating

        self.mu /= cnt

    def train(self):
        for step in range(self.max_iter):
            for user,items in self.rating_data.items():
                for item,rui in items.items():
                    rhat_ui = self.predict(user,item)
                    e_ui = rui-rhat_ui
                    self.bu[user] += self.alpha*(e_ui-self.lmbda*self.bu[user])
                    self.bi[item] += self.alpha*(e_ui-self.lmbda*self.bi[item])
                    # 随机梯度下降更新梯度
                    for k in range(0,self.F):
                        self.P[user][k] += self.alpha*(e_ui*self.Q[item][k] - self.lmbda*self.P[user][k])
                        self.Q[item][k] += self.alpha*(e_ui*self.P[user][k] - self.lmbda*self.Q[item][k])
            self.alpha *= 0.9# 为什么学习率低影响这么大

    def predict(self,user,item):
        return sum(self.P[user][f]*self.Q[item][f] for f in range(0,self.F))+self.bu[user]+self.bi[item]+self.mu

    def predict_bynum(self,user,item):
        matrix_P = pd.DataFrame(self.P).T.values
        matrix_Q = pd.DataFrame(self.Q).T.values
        matrix_bi = pd.DataFrame(self.bi,index=[0]).T.values
        matrix_bu = pd.DataFrame(self.bu,index=[0]).T.values
        # df = pd.DataFrame(matrix_P)
        # df.to_csv('F:\\1120241382\\matrix_P.csv',mode='w')
        # df = pd.DataFrame(matrix_Q.T)
        # df.to_csv('F:\\1120241382\\matrix_Q.csv',mode='w')
        # df = pd.DataFrame(matrix_bi)
        # df.to_csv('F:\\1120241382\\matrix_bi.csv',mode='w')
        # df = pd.DataFrame(matrix_bu)
        # df.to_csv('F:\\1120241382\\matrix_bu.csv',mode='w')
        # print(user)
        # print(matrix_P)
        # for f in range(0, self.F):
        #     print(matrix_P[user][f])
        return sum(matrix_P[user][f]*matrix_Q[item][f] for f in range(0,self.F))+matrix_bu[user]+matrix_bi[item]+self.mu

    def predict_matrix(self):
        metrix = np.zeros((pd.DataFrame(self.P).shape[-1],pd.DataFrame(self.Q).shape[-1]),dtype=float)
        for user in range(pd.DataFrame(self.P).shape[-1]):
            for item in range(pd.DataFrame(self.Q).shape[-1]):
                # print(self.predict_bynum(user,item))
                metrix[user][item] =self.predict_bynum(user,item)
        return metrix

def loadData():
    rating_data = {1: {'A': 5, 'B': 3, 'C': 4, 'D': 4},
                   2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
                   3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
                   4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
                   5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
                   }
    return rating_data

rating_data = loadData()
basicsvd = SVD(rating_data,F = 10,max_iter=100)
basicsvd.train()
metrix = basicsvd.predict_matrix()
print(metrix)
# print(basicsvd.predict_bynum(0,0))
# for item in ['E']:
#     print(item,basicsvd.predict(1,item))