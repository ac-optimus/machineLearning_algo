#contains all the distance metrics
import numpy as np

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
        den = np.dot(x1,x2)   #pointwise dot porduct
        num = np.linalg.norm(x1)*np.linalg.norm(x2)   #product of there norms
        return num/den