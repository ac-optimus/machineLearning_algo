import cvxpy as cp 
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from utils import prepareDatase,accuracy
from sklearn .utils import shuffle
import matplotlib.pyplot as plt

#let y_train be a column matrix
#let x_train be a nxm matrix where n is the number of instance and m is the number of features
class svm_utils:
    def get_constraint(self): 
        #w,y are column ndarrays ,
     #   print (np.array(y_train).shape, np.matrix(self.x_train).shape)
        constrn = np.matrix(self.x_train)@self.w + self.b
#        constrn =  np.multiply(np.matrix(y_train),constrn
        constrn = cp.mul_elemwise(np.array(y_train), constrn)
        return [constrn >= 1]

    def get_objective(self):
        return cp.Minimize(cp.norm(self.w,2)*(1/2))

    def hard_svm(self):
        objective = self.get_objective()
        constraints = self.get_constraint()
        prob = cp.Problem(objective, constraints)
        
        prob.solve()
        self.dual_values = constraints[0].dual_value
        #w changes in place
    def get_dual_values(self):
        alpha = []
        for i in self.dual_values.tolist():
            alpha.append(i[0])
        return alpha
        
    def svm_class(self, y,code = False, Decode = False):
        #replace the two classes by +1 and -1
        if code == True:
            classes = np.unique(y).tolist()    
            d  = {classes[0]:1,classes[-1]:-1}
            self.class_ = d
            y[y.columns[0]] = y[y.columns[0]].map(d)
        elif Decode == True:
            y = list(y)
            classes = list(self.class_.keys())
            for i in range(len(y)):
                if y[i] == 1:
                    y[i] = classes[0]
                elif y[i] == -1:
                    y[i] = classes[-1]
        return y
        
class SVM(svm_utils):
    def __init__(self,type = "hard",b = -11, size_w = 2,  ):
        super().__init__()
        self.b = cp.Variable(1)
        self.w = cp.Variable(size_w)

    def fit(self,x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        #get the optimal hyperplane here
        y_train = self.svm_class(y_train, code = True).values
        self.hard_svm() #ge the hyperplane 
        
    def predict(self, x_test):
        #we use the updated w here
        pred = np.matrix(x_test)@self.w.value+self.b.value
        pred = np.sign(pred)
        pred_class = self.svm_class(pred, Decode = True)
        return pred
    def return_supportVec(self):
        supVec = {1:[],-1:[]}
        for i in range(len(x_train)):
            y_hat = np.matrix(x_test)@self.w.value+self.b.value
            if y_hat == 1.0:
                supVec[1].append(i)
            elif y_hat == -1.0:
                supVec[-1].append(i)
        return supVec
                