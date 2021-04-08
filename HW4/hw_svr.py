import numpy as np
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
import pandas as pd
import operator


class Polynomial:

    def __init__(self, M):
        self.M = M

    def __call__(self, X,XX):
        res = (1+X.dot(XX.T))**self.M
        return res


class SVR:

    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def fit(self,X,y):
        K = self.kernel(X,X)
        x = np.repeat(np.repeat(K, 2).reshape(tuple(map(lambda x, y: x * y, (K.shape[1], K.shape[0]), (1,2))) ), 2, axis=0)
        z = np.array([[1, -1], [-1, 1]])
        z = np.tile(z, K.shape)
        P = matrix( np.multiply(x,z) )

        Y = np.repeat(y.T[0],2)
        one = np.tile([1,-1], len(y))
        q = matrix( np.array([np.multiply(Y,one) - self.epsilon]).T )

        A = matrix((np.array([one*1.])))

        a = np.diag(np.ones(len(y)*2))
        b = -1*np.diag(np.ones(len(y)*2))
        G = matrix ( np.concatenate((a,b)) )

        h = matrix( np.repeat([1/self.lambda_, 0], len(y)*2) )

        b = matrix(0.)

        res = solvers.qp(P=P*1.,q=q,G=G,h=h,A=A,b=b)
        alphas = np.array(res['x'])

        return SVRModel(self.kernel, X, alphas, 0)

class SVRModel:

    def __init__(self, kernel, X, alphas, b ):
        self.kernel = kernel
        self.X = X
        self.aphas = alphas
        self.b = b

    def predict(self, X):


def normalize(X):
    up = (X - np.mean(X, axis=0))
    down = np.std(X, axis=0)
    return up / down


if __name__ == '__main__':
    sine = pd.read_csv('sine.csv', sep=',')
    sine_x = sine['x'].values
    sine_y = sine['y'].values

    new_data = np.arange(1, 20, step=0.2)

    sine_x = sine_x.reshape((sine_x.shape[0], 1))
    sine_y = sine_y.reshape((sine_y.shape[0], 1))
    new_x = new_data.reshape((new_data.shape[0], 1))

    fitter = SVR(kernel=Polynomial(M=11), lambda_=0.01, epsilon = 0.5)
    m = fitter.fit(normalize(sine_x), sine_y)

