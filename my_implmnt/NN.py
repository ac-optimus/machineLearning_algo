


class activation_funcitons:
    #TODO
    #add other activation functions
    def sigmoid(self, z):
        num = np.exp(z)
        den = 1 + np.exp(z)
        return num/den
    def ReLU(self, z):
        return max(0, z)

class cost_Fun:
    #TODO 
    #add other cost functions
    def 
        
class NN(activation_funcitons):
    def __init__(self):
        pass

    def get_activation(self,name,x):
        if name == "sigmoid":
            return self.sigmoid(x)

    def Model(self, inpt_siz, hiddenLayerLst, out_siz):
        #randomly initilaized weights
        w_i, b_i = 0, 0 
        j = inpt_siz
        for i in hiddenLayerLst: #hiddenLayerLst does not contain output layer in it
            w_i = np.random.rand(j,i)
            b_i = np.random.rand(1,i)
            j = i
            self.weights.append(w_i)
            self.b.append(b_i)
        w_i = np.random.rand(j,output)
        b_i = np.random.rand(1,output)
        self.weights.append(w_i)
  
    def buildNN(self,inpt_siz, hiddenLayerLst, activation_lst, out_siz):
        self.weight = self.initialize_weight(inpt_siz, hiddenLayerLst, out_siz)
        self.activation = activation_lst


    def train (self,X):
        #TODO

        A_out = self.forward(X)
        self.backpropogation(loss)

    def backpropogation(self, loss, X):  #loss is a function
        A_out = self.forward(X)

    def getW(self,W_i):

    def forward(self,X):
'''
        Z_1 = A_0*w_1 +b_1
        A_1 = ACTIVATION(Z_1)
        
        Z_2 = A_1*w_2 +b_2
        A_2 = ACTIVATION(Z_2)
        
        Z_3 = A_2*w_3 + b_3
        A_3 = ACTIVATION(Z_3)

'''
        weight = self.weight
        bias = self.b
        Z_i = 0
        A_i = X[0,:] #assign the fist row as it is
        for i in range(len(weight)):
            Z_i = A_i*weight[i] + bias[i]
            A_i = self.get_activation(self.activation_lst[i], Z_i)  #returns a string
        A_i = output
        return output




if __name__ == "__main__":
    X = np.random.rand(100,20)
    test_set = np.random.rand(50,20)
    inputCnt = X.size
    outputCnt = 1  
    lst_hidden = [4,2] #2 hidden layers- first one has 4 neurons and second layer has 2 neurons
    lst_cost = ['ReLU','ReLU','Linear']
    my_ntwork = NN()
    my_ntwork.Model(inputCnt, outputCnt, lst_hidden, lst_cost)
    my_ntwork.train(X)
    pred = my_ntwork.forwar(test_set)

    