import pandas as pd
import numpy as np 
from utils import prepareDatase
from sklearn.utils import shuffle

class GaussianNaiveBayes_utils:

    def var_classWise(self, datafrm):
        return datafrm.groupby(datafrm.columns[-1]).var()
        
    def mean_classWise(self, datafrm):
        return datafrm.groupby(datafrm.columns[-1]).mean()

    def probDist(self,x,mean_ifeatur,var_ifeatur):
        num = np.exp((np.negative(np.square(x-mean_ifeatur)))\
                     /(2*np.square(var_ifeatur)))
        den = np.sqrt(2*np.pi*np.square(var_ifeatur))
        return num/den
    
    def class_prob(self):  
        sum_classes = self.df[self.df.columns[-1]].value_counts(normalize= True)
        self.classProb = {}
        for i in self.classes:
            self.classProb[i] = sum_classes[i]
        return self.classProb

    def prod_prob_fet_given_class_k(self,x,k):
        #x ---> fet1 fet2 we have to predict the class
        fet_prob = 1
        for j in x.columns:
            mean_j, var_j = self.df_mean[j][k], self.df_var[j][k]
            fet_prob *= self.probDist(x[j].values[0],mean_j,var_j)
        return fet_prob    
    
    def probX_given_class_k(self,x,k):               
        #return P(k_class|x_testInstance)
        num = self.classProb[k]*self.prod_prob_fet_given_class_k(x,k)
        den = 1  #no need to evaluate so just let it be some constant
        return num/den

class GaussianNaiveBayes(GaussianNaiveBayes_utils):
    def __init__(self):
            super().__init__()
    
    def fit (self, x_train, y_train):
        self.df  = x_train.join(y_train)
        self.df_mean= self.mean_classWise(self.df)
        self.df_var = self.var_classWise(self.df)
        self.classes = self.df_mean.index.tolist()
        self.classProb = self.class_prob()

    def predict(self,x):
        y_hat = []
        for xi in range(len(x)):
            class_score=[]
            for i in self.classes:
                class_score.append(self.probX_given_class_k(x[xi:xi+1],i))
#             print (class_score)
            y_hat.append(self.classes[np.argmax(class_score)])
        return y_hat
    

    