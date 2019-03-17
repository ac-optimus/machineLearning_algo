#class for knn classification and regression task
from utils import dist_metic
from evaluationMetric import eval
import numpy as np
from sklearn.metrics import mean_squared_error

    
class knn_helper:
    def _select_dist(self,x1,x2):
        if self.distance_metric == "Euclidean":
            return self.__Euclidn__(x1, x2)
        elif self.distance_metric == "Cosine-Similarity":
            return self.__Cosn__(x1, x2)
        elif self.distance_metric == "Manhattan":
            return self.__Manhattn__(x1, x2)
        elif self.distance_metric == "no_numpy_euclidin":
            return self._Euclidn_noNumpy(x1,x2)        

    def prdct_forPnt(self,k_nrstPnts):
        k_nrstPnts = list(k_nrstPnts)
        if self.type == "Classification":
            #perform classification
            return max(k_nrstPnts,key=k_nrstPnts.count)
            
        elif self.type == "Regression":
            #perform regression, we do interpolation
            return sum(k_nrstPnts)/len(k_nrstPnts)
    def get_dstnces(self, pnt):
        pnt_distVs_y_val = {}
        for j in range(len(self.x_train)):
            l = self._select_dist(pnt, self.x_train[j])
            pnt_distVs_y_val[l] = self.y_train[j]
        return pnt_distVs_y_val
    
    def get_kNN(self, DistLst):
        k_nearstPnts_y_val = []
        for j in range(self.k):
            k_nearstPnts_y_val.append(DistLst[j])
        return k_nearstPnts_y_val

class KNN(dist_metic,knn_helper):
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
