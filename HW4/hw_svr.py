import numpy as np
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
import pandas as pd
import operator
import matplotlib.pyplot as plt
import math


class Polynomial:

    def __init__(self, M):
        self.M = M

    def __call__(self, X,XX):
        res = (1+X.dot(XX.T))**self.M
        return res

class RBF:

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, XX):
        if X.ndim == 1:
            X = np.array([X])
        if XX.ndim == 1:
            XX = np.array([XX])
        norm = (np.diag(X.dot(X.T)) - 2*X.dot(XX.T).T).T + np.diag(XX.dot(XX.T))
        if norm.shape[0] ==1 or norm.shape[1]==1:
            norm = norm[0]
        return math.e**(-norm/(2*self.sigma)**2)


class SVR:

    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def fit(self,X,y):
        print(y.shape)
        if y.ndim == 1:
            y = np.array([y]).T
        print(y.shape)

        K = self.kernel(X,X)
        x = np.repeat(np.repeat(K, 2).reshape(tuple(map(lambda x, y: x * y, K.shape, (1,2))) ), 2, axis=0)
        z = np.array([[1, -1], [-1, 1]])
        z = np.tile(z, K.shape)
        P = matrix( np.multiply(x,z) )

        Y = np.repeat(y.T[0],2)
        one = np.tile([1,-1], len(y))
        q = matrix( -np.array([np.multiply(Y,one) - self.epsilon]).T )

        A = matrix((np.array([one*1.])))

        a = np.diag(np.ones(len(y)*2))
        b = -1*np.diag(np.ones(len(y)*2))
        G = matrix ( np.concatenate((a,b)) )

        h = matrix( np.repeat([1/self.lambda_, 0], len(y)*2) )

        b = matrix(0.)

        res = solvers.qp(P=P*1.,q=q,G=G,h=h,A=A,b=b)
        alphas = np.array(res['x'])
        #print(alphas)
        alphas = np.reshape(alphas,(len(X),2))
        print(alphas.shape)

        #epsilon okolica
        e1 = 1e-5
        e2 = 1/self.lambda_ - e1


        #we got the alphas, now we need to calculate b
        vector_diff = np.dot(alphas, np.array([1, -1]))
        w = np.array([np.dot(K, vector_diff)]).T
        #take only if alphai < C or alphai*>0
        indekses1 = alphas < e2
        #print(indekses1.shape)
        indekses2 = alphas > e1
        indekses_final1 = np.logical_or( indekses1[:,1] , indekses2[:,0] )
        t1 = -self.epsilon+y-w
        # take only if alphai > 0 or alphai*<C
        indekses3 = alphas > e1
        indekses4 = alphas < e2
        indekses_final2 = np.logical_or(indekses3[:, 1], indekses4[:, 0])
        t2 = self.epsilon+y-w
        #print(t1)
        lower = max(t1[indekses_final1])
        upper = min(t2[indekses_final2])
        print("LOWER, UPPER")
        print(lower,upper)



        return SVRModel(self.kernel, X, alphas, (lower[0]+upper[0])/2)

class SVRModel:

    def __init__(self, kernel, X, alphas, b ):
        self.kernel = kernel
        self.X = X
        self.alphas = alphas
        self.b = b

    def predict(self, X):
        K = self.kernel(self.X,X)
        vector_diff = np.dot( self.alphas,np.array([1,-1]) )
        return np.dot(K.T, vector_diff) + self.b

    def get_alpha(self):
        return self.alphas

    def get_b(self):
        return self.b


def normalize(X):
    up = (X - np.mean(X, axis=0))
    down = np.std(X, axis=0)
    return up / down

def sine_plot():
    sine = pd.read_csv('sine.csv', sep=',')
    sine_x = sine['x'].values
    sine_y = sine['y'].values

    new_data = np.arange(1, 20, step=0.1)

    X = sine_x.reshape((sine_x.shape[0], 1))
    y = sine_y.reshape((sine_y.shape[0], 1))
    print(y.shape)
    new_x = new_data.reshape((new_data.shape[0], 1))

    kernels = [Polynomial(11), RBF(0.2)]
    names = ['Polynomial kernel', 'RBF kernel']
    colors = ['r', 'g']

    for i in range(2):
        k = kernels[i]
        reg = SVR(k,0.1,0.5)
        model = reg.fit(normalize(X), y)
        predictions = model.predict(normalize(new_x))
        list1, list2 = zip(*sorted(zip(new_x, predictions)))
        plt.plot(list1, list2, label=names[i], color=colors[i])

    #print(X)
    #print(y)
    plt.scatter(sine_x,sine_y, color='b', label='data  points', alpha=0.2)
    plt.legend()
    plt.savefig("sine.png")
    plt.show()


if __name__ == '__main__':
    sine = pd.read_csv('sine.csv', sep=',')
    sine_x = sine['x'].values
    sine_y = sine['y'].values

    new_data = np.arange(1, 20, step=0.2)
    fig, ax = plt.subplots(1)
    ax.plot(sine_x, sine_y, 'bo', label='Original data')

    sine_x = sine_x.reshape((sine_x.shape[0], 1))
    sine_y = sine_y.reshape((sine_y.shape[0], 1))
    new_x = new_data.reshape((new_data.shape[0], 1))

    fitter = SVR(kernel=RBF(sigma=0.3), lambda_=1, epsilon=0.2)
    m = fitter.fit(normalize(sine_x), sine_y)
    pred = m.predict(normalize(new_x))
    list1, list2 = zip(*sorted(zip(new_x, pred)))
    ax.plot(list1, list2, '-', label='RBF sigma=0.3')
    plt.show()


    sine_plot()

