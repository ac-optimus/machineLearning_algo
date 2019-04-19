import numpy as np


class activation_funcitons:
    #TODO
    #add other activation functions
    def sigmoid(self, z):
        num = np.exp(z)
        den = 1 + np.exp(z)
        return num/den
    def ReLU(self, z):
        return np.maximum(0, z)
'''
class cost_Fun:
    #TODO 
    #add other cost functions
    def 
'''        
class NN(activation_funcitons):
    def __init__(self):
        self.weights = []
        self.b = []

    def get_activation(self,name,x):
        if name == "sigmoid":
            return self.sigmoid(x)
        elif name == "ReLU":
            return self.ReLU(x)
        elif name == "Linear":
            return x

    def initialize_weight(self, inpt_siz, hiddenLayerLst, out_siz):
        #randomly initilaized weights
        w_i, b_i = 0, 0 
        j = inpt_siz
        for i in hiddenLayerLst: #hiddenLayerLst does not contain output layer in it
            w_i = np.random.rand(j,i)
            b_i = np.random.rand(1,i)
            j = i
            self.weights.append(w_i)
            self.b.append(b_i)
            
        w_i = np.random.rand(j,out_siz)
        b_i = np.random.rand(1,out_siz)
        self.weights.append(w_i)
        self.b.append(b_i)
  
    def buildNN(self,inpt_siz, out_siz, hiddenLayerLst, activation_lst):
        self.initialize_weight(inpt_siz, hiddenLayerLst, out_siz)
        print ("weights initialized")
        self.print_networkDIm()
        self.activation = activation_lst


    def train (self,X):
        #TODO

        A_out = self.forward(X)
        print (A_out)
     #   self.backpropogation(loss)

    def backpropogation(self, loss, X):  #loss is a function
        A_out = self.forward(X)

   # def getW(self,W_i):
    def print_networkDIm(self):
        counter = 1
        print ("Hidden Layers:")
        for i,j in zip(self.weights, self.b):
            print ("layer:",counter,"wegiht-",i.shape, "bias-",j.shape)
            counter+=1

    def forward(self,X):

        # Z_1 = A_0*w_1 +b_1
        # A_1 = ACTIVATION(Z_1)
        
        # Z_2 = A_1*w_2 +b_2
        # A_2 = ACTIVATION(Z_2)
        
        # Z_3 = A_2*w_3 + b_3
        # A_3 = ACTIVATION(Z_3)

        weight = self.weights
        bias = self.b
        Z_i = 0
        A_i = X[0,:] #assign the fist row as it is
        for i in range(len(weight)):
            Z_i = A_i@weight[i] + bias[i]
            A_i = self.get_activation(self.activation[i], Z_i)  #returns a string
            output = A_i
        return output




if __name__ == "__main__":
    X = np.random.rand(100,3)
    inputCnt = X.shape[1]
    outputCnt = 1  
    lst_hidden = [4,2] #2 hidden layers- first one has 4 neurons and second layer has 2 neurons
    lst_cost = ['ReLU','ReLU','Linear']
    my_ntwork = NN()
    my_ntwork.buildNN(inputCnt, outputCnt, lst_hidden, lst_cost)
    my_ntwork.train(X)
    #pred = my_ntwork.forwar(test_set)

    