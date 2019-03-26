'''
basic task 
- train the model on seed trains set.
- input the pool dataset, and the query out of it
- add the query entries to the new train set
- train the model on this.
- itterate the above for the number of steps required.


'''
import numpy as np
import pandas as pd
class activeLearning:
    def __init__(self,model = None, query_method = "least confidence", scenario = "pool based"):
        self.model = model
        self.query_method = query_method
        self.scenario = scenario
    
    def get_query(self, pool_Set):
        if self.query_method == "least confidence":
            prob_lst = []
            for i in range(len(pool_Set)):
                prob_i = self.model.predict_proba(pool_Set[pool_Set.columns[:-1]].iloc[i:i+1])
                prob_lst.append(prob_i)
            query_i = np.argmax(1- np.array(prob_lst))
            return pool_Set.iloc[query_i:query_i+1]
            #returns a dataframe instead of series

    def seed_fit(self, x_train, y_train):
        self.seed_x_train = x_train
        self.seed_y_train = y_train
        self.model.fit(x_train,y_train)
    
   # def random_query(self,pool_set):
        

    def add_query(self,query, pool_set):
        x_train = self.seed_x_train.append(query[query.columns[:-1]]) 
        y_train = self.seed_y_train.append(query[[query.columns[-1]]])
        index_query = query.index.item()
        pool_set = pool_set.drop(index_query)
        return pool_set, x_train, y_train

    def pool_based(self, itterate, pool_set):
        #check argmax(1-P(c_k|x))
        for i in range(itterate):
            new_query = self.get_query(pool_set)
            pool_set, x_train_i, y_train_i = self.add_query(new_query,pool_set)
            self.model.fit(x_train_i, y_train_i)
