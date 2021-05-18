
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
        #weight0 = np.random.normal(size=(np.prod(layer_size) + 2*(len(layer_size)-1)))
        weights0 = [ 0.81217268, -0.30587821, -0.26408588, -0.53648431,  0.43270381, -1.15076935, 0.87240588, -0.38060345,  0.15951955, -0.12468519,  0.73105397, -1.03007035]
        print(f'Weights {weights0}')
        '''
        weights0 = []
        for i in range(len(layer_size)-1):
            temp = np.random.normal(size=(layer_size[i+1], layer_size[i]))
            weights0.append(temp)'''
        #print('weights')
        #print(weights0)


        #weights_test = np.array([np.array([[0.1,0.3],[0.2,0.4]]), np.array([[0.4,0.2],[0.3,0.1]])])
        #grid(weights_test, layer_size)

        #print(feed_forward(weights0,X, layer_size))
        #print(cost_function(weights0,X,y,layer_size))

        #print(gradient(weights0,X,y,layer_size))

        def f(w) : return cost_function(w,X,y, layer_size)
        def df(w) : return gradient(w,X,y,layer_size)
        #print(f'############ {weights_test.shape}')
        print(test_gradient(f,df , layer_size))

        #weights_opt, _, _ = fmin_l_bfgs_b(cost_function, weights0, fprime=gradient, args=(X, y), maxfun=1e6, maxiter=1e3, factr=1e9)

def grid(weights, layers):
    '''
    grad_flatten = []
    for i in weights:
        grad_flatten += list(i.flatten())
    weights = grad_flatten'''
    print(layers)
    grid = []
    for i in range(len(layers)-1):
        try:
            m, n = layers[i]+1, layers[i+1]
            w = weights[:m*n]
            grid.append(np.reshape(w,(m,n)))
            weights = weights[m*n:]
        except:
            print('##############3')
    u = 3
    return grid



def cost_function(weights,X,y, layers):
    #weights = grid(weights, layers)
    #a = np.dot(weights,X.T)
    #print(f'a : {a}')
    #pred = sigmoid(a)
    a = feed_forward(weights,X,layers)[-1]
    #pred = sigmoid(a)
    #last = softmax(pred) #DO WE NEED TO USE SOFTMAX AT LAST STEP?
    return log_loss(a,y, 2 )

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
    temp = (0,num_classs-1)
    losses = np.log(pred[temp,y])
    return -np.sum(losses) /num_classs #ALI JE TREBA TU DELIT Z DOLŽINO KLASOV?

def softmax(x):
    norm = np.sum(np.exp(x), axis=1, keepdims=True)
    return np.exp(x) / norm

def feed_forward(weights,X, layers):
    weights = grid(weights, layers)
    print(weights)
    #ones = np.array([np.ones(X.shape[0])]).T
    #a = np.append(X, ones, axis = 1) #add ones for summing the biases

    ones = np.array([np.ones(X.shape[0])]).T
    a=np.append(X, ones, axis=1)
    a_list = [a]
    for weight in weights[:-1]:
        print(f'a : {a}')
        a = np.array(sigmoid(a.dot(weight)))
        ones = np.array([np.ones(a.shape[0])]).T
        a = np.append(a, ones, axis=1)
        #a = np.dot( a,weight.T)
        #a= a.dot(weight)
        a_list.append(a)


        print(f'aa:{a}')
    #ones = np.array([np.ones(a_list[-1].shape[0])]).T
    #a = np.append(a_list[-1], ones, axis=1)
    a_list.append(softmax(a.dot(weights[-1])))
    return a_list


def gradient(weights,X,y, layers):
    a = feed_forward(weights,X, layers)
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
    '''
    for i in range(1,len(weights)+1):
        if i == 1:
            current_weights = weights[-1] #we start from the back
            current_activations = a[-1]
            next_activations = a[-i-1]
            temp = np.zeros(num_classes)
            temp[y[0]] = 1
            temp = np.array([temp]).T
            t = np.multiply(-1/sigmoid(current_activations), temp)
            activations_multy = np.squeeze([next_activations for _ in range(len(current_weights))]).T
            der = inv_sigmoid(np.dot(current_weights,activations_multy.T))[0]
            delta =  np.multiply(t,np.multiply(der,next_activations))
            deltas.append(delta)
        else:
            current_weights = weights[-i]  # we start from the back
            current_activations = a[-i]
            next_activations = a[-i - 1]
            previous_delta = deltas[-i+1].T
            #activations_multy = np.squeeze([next_activations for _ in range(len(current_weights))]).T
            #der = inv_sigmoid(np.dot(current_weights, activations_multy))[0]*previous_delta
            t = inv_sigmoid(current_activations).T.dot(next_activations)
            delta = previous_delta.dot(t)
            #delta = np.multiply(der, next_activations)

            deltas.append(delta)'''

    weights = grid(weights, layers)

    '''
    #last layer - easier
    ti = np.zeros(shape=(a[-1].shape[0], layers[-1]))
    ti[(0,a[-1].shape[0]-1), y] = 1
    delta = a[-1]-ti
    deltas.append(a[-2].T.dot(delta))

    #the rest
    for i in range(1,len(weights)):
        hj = np.delete(a[-i-1]* 1-a[-i-1], -1, axis = 1)
        weights_i = np.delete(weights[-i], -1, axis=0).T
        delta=(hj * deltas[i-1].dot(weights_i))
        deltas.append(a[- i -2].T.dot(delta))

    deltas.reverse()
    return deltas'''
    grad = [np.zeros(w.shape) for w in weights]
    delta = [[] for _ in weights]

    if True:
        delta[-1] = a[-1]
        delta[-1][range(len(y)), y] -= 1
        # delta[-1] /= len(y)
        grad[-1] = a[-2].T.dot(delta[-1])
    for i in range(1, len(weights)):
        act_i = np.delete(a[-1 - i] * (1 - a[-1 - i]), -1, axis=1)
        weights_i = np.delete(weights[-i], -1, axis=0).T
        delta[-1 - i] = act_i * delta[-i].dot(weights_i)
        grad[-1 - i] = a[-2 - i].T.dot(delta[-1 - i])
    return grad


def test_gradient(f,df, layers):
    eps = 1e-10
    tol = 1e-3
    n = np.prod(layers) + sum([i for i in layers[:-1]]) #weights + bias
    print(n)
    for _ in range(100):
        #x0 = np.random.normal(size=n)
        x0 = [ 0.81217268, -0.30587821, -0.26408588, -0.53648431,  0.43270381, -1.15076935, 0.87240588, -0.38060345,  0.15951955, -0.12468519,  0.73105397, -1.03007035]

        #x0 = np.array([0.2,0.3,0.1,0.1])
        #x0 = grid([x0], layers )
        #x0 = np.array([[0.2,0.1], [0.3,0.1]])
        #x0 = np.array([np.array([[0.1, 0.3], [0.2, 0.4]]), np.array([[0.4, 0.2], [0.3, 0.1]])])
        #x0 = np.array(x0, dtype=object)
        #x0 = np.array([np.array([np.random.normal(size=2), np.random.normal(size=2)])],dtype=object)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = eps
            #e_i = grid(e_i, layers)
            grad = df(x0)
            grad_flatten = []

            #e_i = np.array(e_i, dtype=object)
            for j in grad:
                grad_flatten += list(j.flatten())
            grad_flatten = np.array(grad_flatten)
            try:
                fplus = f(np.add(x0 , e_i))
            except:
                a=3
            fminus = f(x0)
            if (fplus - fminus) / (eps)  - grad_flatten[i]*0.5 > tol:
                a=3
                return False
            else:
                print('|||||||||||||||||||||||||||||||||||||||||||||||||||||')


    return True



if __name__ == '__main__':
    np.random.seed(1)
    '''
    X = pd.read_csv('housing3.csv')[:1][['CRIM', 'INDUS']].to_numpy()
    y = pd.read_csv('housing3.csv')[:1][['Class']].to_numpy()
    y = np.asarray(y=='C1', dtype = int)
    '''
    X = np.array([[1, 0.3], [0.5, 0.6]])
    y = np.array([0, 1])

    ann = ANNClassification([2])

    ann.fit(X,y)
    #print(inv_sigmoid(np.array([0.8])))

    #print(log_loss(np.array([[0.2],[0.5],[0.3]]),np.array([[1],[0],[0]]).T))
    #print(sigmoid(np.array([[1],[2],[3]])))

    #weights0 = np.random.normal(size=(2,2))
    #print(gradient(X, y, weights0))