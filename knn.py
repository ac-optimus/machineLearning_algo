#class for knn classification and regression task
from distanceMetric import dist_metic
from evaluationMetric import eval
from knn_helper import helper
import numpy as np
from sklearn.metrics import mean_squared_error

    
class KNN(dist_metic,helper):
    #we will be playing around with matrix here
    def __init__(self, type_ = 'Classification',\
                     x_train=None, y_train=None, \
                        x_test=None,y_test=None, \
                            distance_metric = 'Euclidean', K=3):
        
        self.k = K
        self.type = type_
        self.distance_metric = distance_metric   
   
    def predict(self,  x_test):
        x_test = np.matrix(x_test)
        y_hat = np.zeros(len(x_test))

        for xi in range(len(x_test)):               
            dist_order = list(self.get_dstnces(x_test[xi]))#list(some_dict) gives the list of keys
            np.sort(dist_order)    #inplace sort
            kNN = self.get_kNN(dist_order)#pass the distances
            y_hat[xi]=self.prdct_forPnt(kNN)         
        return y_hat

    def fit(self,x,y):
        self.x_train = np.matrix(x)
        self.y_train = np.array(y)  
       
    def test_(self,y_hat,ytest,eval_measure = "RMSE"):
        y = ytest
        try:
            if eval_measure == "RMSE":
                return np.sqrt(mean_squared_error(y_hat,y))
       # except Exception as e :
        except AttributeError:
            return "first call predict method."
