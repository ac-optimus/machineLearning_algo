import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing
from sklearn.utils import shuffle

def accuracy(y_hat,y):
    score = 0
    for i in range(len(y_hat)):
        if y_hat[i] == y[i]:
            score += 1
        
    return score/len(y_hat)


def evaluate_Poly(X_values,theata): 
    value = np.matmul(X_values,theata[1:]) + theata[0]
    return value  #returns a column

def residual(x,y,theata): # x is row matrix and theata is column mat
    error = y - evaluate_Poly(x,theata)
    return error

def get_n_features(dataset,k):
    #returns all possible combination of k features out of all the features present
    return list(itertools.combinations(dataset.columns,k))

def feature_polynomial(seed_dataset,d):  #dataset is a list of 
    feature_list = []
    index = []
    for i in range(1,d+1):
        index.append('x'+str(i))
        feature_i = seed_dataset**i
        feature_list.append(feature_i)
    feature_list = np.array(feature_list).transpose()
    new_Dataset = pd.DataFrame(list(feature_list),columns = index)
    return new_Dataset


class dist_metic:

    def __Euclidn__(self,x1,x2):   #x1 and x2 can be matrix
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.sqrt(np.sum(np.square(np.subtract(x2,x1))))
   
    def __Manhattn__(self,x1,x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.sum(np.absolute(np.subtract(x2,x1)))

    def __Cosn__(self,x1,x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        den = np.dot(x1,x2.transpose()).item()   #pointwise dot porduct
        num = np.linalg.norm(x1)*np.linalg.norm(x2)   #product of there norms
        return den/num
    def _Euclidn_noNumpy(self,x1,x2):  
        x1 = np.array(x1)
        x2 = np.array(x2)
        dist = 0
        for i in range(len(x1)):
            dist += (x1[i] - x2[i])**2
        
        return sum(dist)**(0.5)


class prepareDatase:

    def __repr__(self):
        return "Parameter : dataset(.csv),normalized,ratio of train to test to split the dataset,split the dataset into (x_train, y_train, x_test, y_test)."\
            "\n Methods: \n noramilize_theDatst : Normalize dataset feature wise.\n"\
                " splitInto_test_train : split dataset into given feature\n get_train_test : returns train and test set queries."
    def noramilize_theDatst(self, dataset):
        normalzd_datast = dataset.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        new_ = min_max_scaler.fit_transform(normalzd_datast)
        normalzd_datast = pd.DataFrame(new_)
        d,p = {},{}
        for i,j in  zip(dataset.columns,normalzd_datast.columns):
            d[j] = i
        for i,j in zip(dataset.index,normalzd_datast.index):
            p[j] = i
        return normalzd_datast.rename(columns = d,index=p)

    def splitInto_test_train(self,dataset,ratio):
        trainset = dataset.iloc[:int(ratio*len(dataset.index))]
        testset = dataset.iloc[int(ratio*len(dataset.index)):]
        return trainset,testset

    def get_train_test(self,dset,normlzd = False,\
                                ratio = 1,x_y_split = False):
            
        dataset = pd.read_csv(dset)
        dataset = dataset[dataset.columns[1:]]
        dataset = shuffle(dataset)
        #test:train == 30%:70% dataset
        if normlzd == True:
            dataset = self.noramilize_theDatst(dataset)
        train,test = self.splitInto_test_train(dataset,ratio)  

        if x_y_split == False:
            return train,test
        else:
            x_train = train[train.columns[:-1]]
            y_train = train[[train.columns[-1]]]
            x_test  = test[test.columns[:-1]]
            y_test = test[[test.columns[-1]]]          
            return x_train,y_train,x_test,y_test

    def split_into_instances_and_label(self,dataset):
        x_dataset = dataset[dataset.columns[:-1]]
        y_dataset = dataset[[dataset.columns[-1]]]
        return x_dataset, y_dataset
