#class for knn classification and regression task
from distanceMetric import dist_metic
from evaluationMetric import eval
import numpy as np

class KNN(dist_metic,eval):
    #we will be playing around with matrix here
    def __init__(self, type_ = 'Classification',\
                     x_train=None, y_train=None, \
                        x_test=None,y_test=None, \
                            distance_metric = 'Euclidean', K=3):
        
        self.k = K
        self.type = type_
        self.distance_metric = distance_metric
        
    
    def prediction(self, x_train, x_test, y_train, y_test):
        
        x_train,x_test = map(np.matrix ,(x_train,x_test))
        y_train, y_test = map(np.array, (y_train,y_test))
        y_hat = np.zeros(len(y_test))

        for xi in range(len(x_test)):
            pnt_distVs_y_val = {}
            for i in range(len(x_train)):
                l = self._select_dist(x_test[xi], x_train[i])
                pnt_distVs_y_val[l] = y_train[i]
                #distance versus the y_label/y_value
            dist_order = list(pnt_distVs_y_val.keys())
            np.sort(dist_order)    #inplace sort
            k_nearstPnts_y_val = []
            for j in range(self.k):
             #pick the fist k elements
                k_nearstPnts_y_val.append(dist_order[j])
            y_hat[xi]=self.prdct_forPnt(k_nearstPnts_y_val)
        self.y = y_test
        self.y_hat = y_hat
        return y_hat
    
    def _select_dist(self,x1,x2):
        if self.distance_metric == "Euclidean":
            return self.__Euclidn__(x1, x2)
        elif self.distance_metric == "Cosine-Similarity":
            return self.__Cosn__(x1, x2)
        elif self.distance_metric == "Manhattan":
            return self.__Manhattn__(x1, x2)
  

    def prdct_forPnt(self,k_nrstPnts):
        k_nrstPnts = list(k_nrstPnts)
        if self.type == "Classification":
            #perform classification
            return max(k_nrstPnts,key=k_nrstPnts.count)
            
        elif self.type == "Regression":
            #perform regression, we do interpolation
            return sum(k_nrstPnts)/len(k_nrstPnts)
        
    def test_(self,eval_measure = "RMSE"):
        try:
            if eval_measure == "RMSE":
                return self.rmse(self.y_hat,self.y)
       # except Exception as e :
        except AttributeError:
            return "first call predict method."
