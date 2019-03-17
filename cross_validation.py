import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
class cross_validation:
    def __init__(self,
                 model=None,\
                 hyperparam_lst=None):
        self.model = model  #pass model object that needs to be train
        self.hyperparam_lst = hyperparam_lst
        
    def _splt_datst(self,n,data):
        ptr = 0
        split = data.shape[0]/float(n)
        lst_datset = []
        while ptr < data.shape[0]:
            lst_datset.append(data[int(ptr):int(ptr+split)])
            ptr = ptr+split
        return lst_datset
    def optim_hyperparam(self,n,dataset):
        folds = self._splt_datst(n,dataset)
        lst_alpha = []
        for i in range(len(folds)):
            val = folds[i]
            train_i = pd.concat(folds[:i]+folds[i+1:])
            alpha_bst_i = [self.hyperparam_lst[0],np.inf]
            
            for i in self.hyperparam_lst:  #check for best lambda in a fold
                optiml_param  = i
                self.model.n_neighbors = optiml_param
                self.model.fit(train_i[train_i.columns[:-1]],\
                                  train_i[[train_i.columns[-1]]])
                y_pred = self.model.predict(val[val.columns[:-1]])
                mse = mean_squared_error(y_pred,val[val.columns[-1]])
                if mse < alpha_bst_i[1]:
                    alpha_bst_i[0] = optiml_param
                    alpha_bst_i[1] = mse
            lst_alpha.append(alpha_bst_i)
        return lst_alpha

    def report_scores(self,train,test,n):
        folds = self._splt_datst(n,train)
        bst_hyprPrm = self.optim_hyperparam(n,train)
        report = []
        for i in range(len(folds)):
            val_i = folds[i]
            train_i = pd.concat(folds[:i]+folds[i+1:])
            self.model.n_neighbors = bst_hyprPrm[i][0]
            self.model.fit(train_i[train_i.columns[:-1]],\
                                  train_i[[train_i.columns[-1]]])
            y_pred_val = self.model.predict(val_i[val_i.columns[:-1]])
            y_pred_test = self.model.predict(test[test.columns[:-1]])
            y_pred_train = self.model.predict(train[train.columns[:-1]])
            
            mse_val  = mean_squared_error(y_pred_val,val_i[val_i.columns[-1]])
            mse_test = mean_squared_error(y_pred_test,test[test.columns[-1]])
            mse_train = mean_squared_error(y_pred_train,train[train.columns[-1]])
            report.append((self.model.n_neighbors,(mse_train,mse_val,mse_train)))
        return report