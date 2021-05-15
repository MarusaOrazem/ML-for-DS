
import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b


class ANNClassification:

    def __init__(self, units= [], lambda_ = 0.1):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y):
        m = X.shape[1]
        last_layer = len(np.unique(y))
        self.classes = last_layer
        layer_size = [m] + self.units + [2] #CHANGE THIS WHEN Y WILL HAVE ALL CLASSES

        #create matrix of weights
        #weights0 = np.random.normal(size=(layer_size[-1],layer_size[0]))
        weights0 = []
        for i in range(len(layer_size)-1):
            temp = np.random.normal(size=(layer_size[i+1], layer_size[i]))
            weights0.append(temp)
        #print('weights')
        #print(weights0)


        weights_test = [np.array([[0.1,0.3],[0.2,0.4]]), np.array([[0.4,0.2],[0.2,0.1]])]

        print(feed_forward(weights_test,X, layer_size))

        #print(gradient(weights_test,X,y,2))

        def f(w) : return cost_function(w,X,y)
        def df(w) : return gradient(w,X,y,weights0.shape[0])
        #print(f'############ {weights_test.shape}')
        #print(test_gradient(f,df,weights0.shape))

        #weights_opt, _, _ = fmin_l_bfgs_b(cost_function, weights0, fprime=gradient, args=(X, y), maxfun=1e6, maxiter=1e3, factr=1e9)

def cost_function(weights,X,y):
    a = np.dot(weights,X.T)
    #print(f'a : {a}')
    pred = sigmoid(a)
    #last = softmax(pred) #DO WE NEED TO USE SOFTMAX AT LAST STEP?
    return log_loss(pred,y, weights.shape[0] )

def sigmoid(x):
    #print(1 / (1+np.exp(-x)))
    return 1 / (1+np.exp(-x))

def inv_sigmoid(x):
    #print(f'invers: {np.exp(-x)/(1+np.exp(-x))**2}')
    return np.exp(-x)/(1+np.exp(-x))**2

def log_loss(pred,y, num_classs):
    #vektor 0, na item mestu 1
    temp = np.zeros(num_classs)
    temp[y[0]] = 1

    losses = temp.dot(np.log(pred))
    return -sum(losses)/len(losses)

def softmax(x):
    norm = np.sum(np.exp(x), axis=1, keepdims=True)
    return np.exp(x) / norm

def feed_forward(weights,X, layers):
    a = X.T
    for weight in weights:
        print(f'a : {a}')
        a = np.dot(weight, a)
        a = sigmoid(a)
    return a


def gradient(weights,X,y, num_classes):
    a = feed_forward(weights,X,y)
    t = -1/a
    temp = np.zeros(num_classes)
    temp[y[0]] = 1
    temp = np.array([temp]).T
    t = np.multiply(t,temp)
    diag = np.zeros((X.shape[1],X.shape[1]))
    np.fill_diagonal(diag,X)
    der = inv_sigmoid(np.dot(weights,X.T))
    #print(np.multiply(t,der))
    return 0.5*np.multiply(t,np.dot(der,X)) #CHANGE 0.5 WHEN DIFFERENT SIZES

def test_gradient(f,df,shape):
    eps = 1e-6
    tol = 1e-3
    for _ in range(10):
        count = 0
        x0 = np.random.normal(size=shape)
        #x0 = np.array([[0.2,0.1], [0.3,0.1]])
        for i in range(shape[0]):
            for j in range(shape[1]):
                e_i = np.zeros(shape[0] * shape[1])
                e_i[i+j] = 1
                e_i = np.array([e_i]).reshape(shape)
                grad = df(x0)
                fplus = f(x0 + eps * e_i)
                fminus = f(x0)
                if (fplus - fminus) / eps / 2 - grad[i][j] > tol:
                    return False

    return True



if __name__ == '__main__':

    '''
    X = pd.read_csv('housing3.csv')[:1][['CRIM', 'INDUS']].to_numpy()
    y = pd.read_csv('housing3.csv')[:1][['Class']].to_numpy()
    y = np.asarray(y=='C1', dtype = int)
    '''
    X = np.array([[1]])
    y = np.array([[0]])

    ann = ANNClassification([2])

    ann.fit(X,y)

    #print(log_loss(np.array([[0.2],[0.5],[0.3]]),np.array([[1],[0],[0]]).T))
    #print(sigmoid(np.array([[1],[2],[3]])))

    #weights0 = np.random.normal(size=(2,2))
    #print(gradient(X, y, weights0))