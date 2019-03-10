import numpy as np

class eval:
    def accuracy(self,y_hat,y):
        if y_hat != y:
            return False
        else:
            for i in range(y_hat):
                if y_hat[i] == y[i]:
                    count +=1 
            return count/len(y)   
    def rmse(self,y_hat,y):
        y_hat, y = map(np.array, (y_hat, y))
        return np.sqrt(np.square((np.absolute(np.subtract(y_hat,y)))))