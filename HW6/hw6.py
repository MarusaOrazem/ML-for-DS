
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


        weights_test = [np.array([[0.1,0.3],[0.2,0.4]])]
        #grid(weights_test, layer_size)

        #print(feed_forward(weights_test,X, layer_size))
        #print(cost_function(weights_test,X,y,layer_size))

        #print(gradient(weights_test,X,y,2))

        def f(w) : return cost_function(w,X,y, layer_size)
        def df(w) : return gradient(w,X,y,2)
        #print(f'############ {weights_test.shape}')
        print(test_gradient(f,df,4, layer_size))

        #weights_opt, _, _ = fmin_l_bfgs_b(cost_function, weights0, fprime=gradient, args=(X, y), maxfun=1e6, maxiter=1e3, factr=1e9)

def grid(weights, layers):
    grad_flatten = []
    for i in weights:
        grad_flatten += list(i.flatten())
    weights = grad_flatten
    grid = []
    for i in range(len(layers)-1):
        try:
            m, n = layers[i], layers[i+1]
            w = weights[:m*n]
            grid.append(np.reshape(w,(n,m)))
            weights = weights[m*n:]
        except:
            print('##############3')
    return grid



def cost_function(weights,X,y, layers):
    weights = grid(weights, layers)
    #a = np.dot(weights,X.T)
    #print(f'a : {a}')
    #pred = sigmoid(a)
    a = feed_forward(weights,X,layers)[-1]
    pred = sigmoid(a)
    #last = softmax(pred) #DO WE NEED TO USE SOFTMAX AT LAST STEP?
    return log_loss(pred,y, 2 )

def sigmoid(x):
    #print(1 / (1+np.exp(-x)))
    x = np.array(x,dtype=float)
    return 1 / (1+np.exp(-x))

def inv_sigmoid(x):
    #print(f'invers: {np.exp(-x)/(1+np.exp(-x))**2}')
    x = np.array(x, dtype=float)
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
    a_list = [a]
    for weight in weights:
        #print(f'a : {a}')
        a = np.dot(weight, a)
        a_list.append(a)
        a = sigmoid(a)
    return a_list


def gradient(weights,X,y, num_classes):
    a = feed_forward(weights,X,y)
    '''
    t = -1/a
    temp = np.zeros(num_classes)
    temp[y[0]] = 1
    temp = np.array([temp]).T
    t = np.multiply(t,temp)
    diag = np.zeros((X.shape[1],X.shape[1]))
    np.fill_diagonal(diag,X)
    der = inv_sigmoid(np.dot(weights,X.T))
    #print(np.multiply(t,der))'''
    deltas = []
    for i in range(1,len(weights)+1):
        if i == 1:
            current_weights = weights[-1] #we start from the back
            current_activations = a[-1]
            next_activations = a[-i-1]
            temp = np.zeros(num_classes)
            temp[y[0]] = 1
            temp = np.array([temp]).T
            t = np.multiply(-1/current_activations, temp)
            activations_multy = np.array([next_activations for _ in range(len(current_weights))])
            der = inv_sigmoid(np.dot(current_weights,activations_multy))
            delta = 0.5* np.multiply(t,np.multiply(der,next_activations))
            deltas.append(delta)
        else:
            current_weights = weights[-i]  # we start from the back
            current_activations = a[-i]
            next_activations = a[-i - 1]
            previous_delta = sum(deltas[-i+1])
            der = inv_sigmoid(np.dot(current_weights, next_activations))*previous_delta
            delta = np.multiply(der, next_activations)
            deltas.append(delta)

    deltas.reverse()
    return deltas
    #return 0.5*np.multiply(t,np.dot(der,X)) #CHANGE 0.5 WHEN DIFFERENT SIZES

def test_gradient(f,df,n, layers):
    eps = 1e-6
    tol = 1e-3
    for _ in range(100):
        count = 0
        #x0 = np.random.normal(size=n)
        #x0 = grid([x0], layers )
        #x0 = np.array([[0.2,0.1], [0.3,0.1]])
        #x0 = np.array(x0, dtype=object)
        x0 = np.array([np.array([np.random.normal(size=2), np.random.normal(size=2)])],dtype=object)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = eps
            e_i = grid(e_i, layers)
            grad = df(x0)
            grad_flatten = []

            e_i = np.array(e_i, dtype=object)
            for j in grad:
                grad_flatten += list(j.T.flatten())
            grad_flatten = np.array(grad_flatten)
            try:
                fplus = f(np.add(x0 , e_i))
            except:
                a=3
            fminus = f(x0)
            if (fplus - fminus) / eps / 2 - grad_flatten[i] > tol:
                a=3
                return False
            else:
                print('YES')

    return True



if __name__ == '__main__':

    '''
    X = pd.read_csv('housing3.csv')[:1][['CRIM', 'INDUS']].to_numpy()
    y = pd.read_csv('housing3.csv')[:1][['Class']].to_numpy()
    y = np.asarray(y=='C1', dtype = int)
    '''
    X = np.array([[1,2]])
    y = np.array([[0]])

    ann = ANNClassification()

    ann.fit(X,y)

    #print(log_loss(np.array([[0.2],[0.5],[0.3]]),np.array([[1],[0],[0]]).T))
    #print(sigmoid(np.array([[1],[2],[3]])))

    #weights0 = np.random.normal(size=(2,2))
    #print(gradient(X, y, weights0))