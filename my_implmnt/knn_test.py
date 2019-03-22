#testing out the performance of the knn
import pandas as pd
from sklearn.utils import shuffle
from knn import KNN
import sys
sys.path.append("..")
from utils import dist_metic

#dataset prepration
if __name__ == "__main__":
    df = pd.read_csv("../dataset/realEstate.csv")
    df = shuffle(df/df.mean())#normalize
    train = df.iloc[:2*int(len(df)/3)]#2/3 goes for traning
    test =  df.iloc[2*int(len(df)/3):]
    x_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]]
    x_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]]
    #testing
    my_knn = KNN(type_="Regression",K=5,distance_metric="Cosine-Similarity")
    my_knn.fit(x_train,y_train)
    y_hat = my_knn.predict(x_test)
    #print (y_hat)
    score = my_knn.test_(y_hat,y_test,eval_measure="RMSE")
    print ("RMSE score:",score)