#helper class for my knn
class helper:
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


    