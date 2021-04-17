import numpy as np
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
import pandas as pd
import operator
import matplotlib.pyplot as plt
import math
import random


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
        #print(y.shape)
        if y.ndim == 1:
            y = np.array([y]).T
        #print(y.shape)

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
        #print(alphas.shape)

        #epsilon okolica
        e1 = 1e-2
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
        #print("LOWER, UPPER")
        #print(lower,upper)

        self.vectors = np.where(abs(vector_diff)>e1)



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

    new_data = np.arange(0, 20, step=0.1)

    X = sine_x.reshape((sine_x.shape[0], 1))
    y = sine_y.reshape((sine_y.shape[0], 1))
    print(y.shape)
    new_x = new_data.reshape((new_data.shape[0], 1))

    kernels = [Polynomial(11), RBF(0.2)]
    names = ['Polynomial kernel', 'RBF kernel']
    colors = ['r', 'g']

    for i in range(2):
        k = kernels[i]
        reg = SVR(k,1,0.5)
        model = reg.fit(normalize(X), y)
        predictions = model.predict(normalize(new_x))
        list1, list2 = zip(*sorted(zip(new_x, predictions)))
        plt.plot(list1, list2, label=names[i], color=colors[i])

        vectors = reg.vectors
        plt.scatter(X[vectors], sine_y[vectors], color=colors[i], label='vectors')

    #print(X)
    #print(y)
    plt.scatter(X,sine_y, color='b', label='data  points', alpha=0.2)
    plt.legend()
    plt.savefig("sine.png")
    plt.show()

def RMSE(y1,y2):
    res = 0
    for i in range(len(y1)):
        res += (y1[i]-y2[i])**2
    return np.sqrt(res / len(y1))

def errors_RBF(X,y):
    n = 80 * len(y) // 100
    k = 2
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]

    eps = 8
    # test polynomial kernel
    sigmas = np.arange(0.1, 5, 0.1)
    lambdas = np.arange(0.01, 1, 0.05)
    best_lambdas = []
    support_vectors = []
    for sigma in sigmas:
        final_rmse = []
        for l in lambdas:
            reg = SVR(RBF(sigma), l, eps)
            # we have the regressor, now perform k-fold cross validation and remember each RMSE
            n = len(y_train)

            # create indexes and shuffle them to get elements for k folds
            i_shuffled = [i for i in range(n)]
            random.seed(0)
            random.shuffle(i_shuffled)
            indexes_folds = []
            for i in range(k):
                j = i_shuffled[(i) * n // k: n * (i + 1) // k]
                indexes_folds.append(j)  # save only indexes for each fold

            all_rmse = []
            for i in range(k):
                all_folds = [j for j in range(k)]
                all_folds.remove(i)  # remove the index for test fold

                x_test_cv = X[indexes_folds[i]].reshape((len(indexes_folds[i]), len(X[0])))
                y_test_cv = y[indexes_folds[i]]

                x_train_cv = np.array([])
                y_train_cv = np.array([])
                for j in all_folds:
                    x_train_cv = np.append(x_train_cv, X[indexes_folds[j]])
                    y_train_cv = np.append(y_train_cv, y[indexes_folds[j]])

                x_train_cv = x_train_cv.reshape(((k - 1) * n // k, len(X[0])))
                # fit model and calculate RMSE
                model = reg.fit(x_train_cv, y_train_cv)
                predicted = model.predict(x_test_cv)
                rmse = RMSE(y_test_cv, predicted)
                all_rmse.append(rmse)

            final_rmse.append(np.mean(all_rmse))
        print(final_rmse)
        best_lambdas.append(final_rmse.index(min(final_rmse)))

    best_rmse_cv = []
    # results for constant lambda
    best_rmse_costant = []
    best_support_vectors= []
    for i,sigma in enumerate(sigmas):
        reg = SVR(RBF(sigma), 1, eps)
        model = reg.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rmse = RMSE(y_test, predicted)
        best_rmse_costant.append(rmse)
        vectors = reg.vectors
        best_support_vectors.append(len(vectors[0]))

        reg1 = SVR(RBF(sigma), lambdas[best_lambdas[i]], eps)
        model1 = reg1.fit(x_train, y_train)
        predicted1 = model1.predict(x_test)
        rmse1 = RMSE(y_test, predicted1)
        best_rmse_cv.append(rmse1)
        vectors = reg1.vectors
        support_vectors.append(len(vectors[0]))

    #vectors = reg.vectors
    #plt.scatter(X[vectors], sine_y[vectors], color=colors[i], label='vectors')



    plt.plot(sigmas, best_rmse_costant, color='r', label='Constant lambda = 1')
    plt.plot(sigmas, best_rmse_cv, color='g', label='Best lambda')
    plt.title("RMSE with RBF kernel")
    plt.xlabel("sigma")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("RMSE_RBF.png")

    plt.plot(sigmas, best_support_vectors, color='r', label='Constant lambda = 1')
    plt.plot(sigmas, support_vectors, color='g', label='Best lambda')
    plt.title("Number of support vectors with RBF kernel")
    plt.xlabel("sigma")
    plt.ylabel("number of support vectors")
    plt.legend()
    plt.savefig("RMSE_supp_vec.png")
    plt.show()


def errors_polynomial(X,y):
    n = 80 * len(y) // 100
    k = 2
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]

    # test polynomial kermel
    M = [i for i in range(1, 11)]
    lambdas = np.arange(0.1, 10, 0.2)
    best_lambdas = []
    for m in M:
        final_rmse = []
        for l in lambdas:
            reg = SVR(Polynomial(m), l,2)
            # we have the regressor, now perform k-fold cross validation and remember each RMSE
            n = len(y_train)

            # create indexes and shuffle them to get elements for k folds
            i_shuffled = [i for i in range(n)]
            random.seed(0)
            random.shuffle(i_shuffled)
            indexes_folds = []
            for i in range(k):
                j = i_shuffled[(i) * n // k: n * (i + 1) // k]
                indexes_folds.append(j)  # save only indexes for each fold

            all_rmse = []
            for i in range(k):
                all_folds = [j for j in range(k)]
                all_folds.remove(i)  # remove the index for test fold

                x_test_cv = X[indexes_folds[i]].reshape((len(indexes_folds[i]), len(X[0])))
                y_test_cv = y[indexes_folds[i]]

                x_train_cv = np.array([])
                y_train_cv = np.array([])
                for j in all_folds:
                    x_train_cv = np.append(x_train_cv, X[indexes_folds[j]])
                    y_train_cv = np.append(y_train_cv, y[indexes_folds[j]])

                x_train_cv = x_train_cv.reshape(((k - 1) * n // k, len(X[0])))
                # fit model and calculate RMSE
                model = reg.fit(x_train_cv, y_train_cv)
                predicted = model.predict(x_test_cv)
                rmse = RMSE(y_test_cv, predicted)
                all_rmse.append(rmse)
            final_rmse.append(np.mean(all_rmse))
        #print(final_rmse)
        best_lambdas.append(final_rmse.index(min(final_rmse)))


    best_rmse_cv = []
    # results for constant lambda
    best_rmse_costant = []
    for i, m in enumerate(M):
        reg = SVR(Polynomial(m), 1,2)
        model = reg.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rmse = RMSE(y_test, predicted)
        best_rmse_costant.append(rmse)

        #print(lambdas[best_lambdas[i]])
        reg1 = SVR(Polynomial(m), lambdas[best_lambdas[i]],2)
        model1 = reg1.fit(x_train, y_train)
        predicted1 = model1.predict(x_test)
        rmse1 = RMSE(y_test, predicted1)
        best_rmse_cv.append(rmse1)


    plt.plot(M, best_rmse_costant, color='r', label='Constant lambda = 1')
    plt.plot(M, best_rmse_cv, color='g', label='Best lambda')
    plt.title("RMSE with Polynomial kernel")
    plt.xlabel("M")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("RMSE_polynomial.png")
    plt.show()


if __name__ == '__main__':

    if False:
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

    if True:
        sine = pd.read_csv('housing2r.csv', sep=',')
        sine_x = sine[['RM','AGE','DIS','RAD','TAX']].values
        sine_y = sine['y'].values

        #sine_x = sine_x.reshape((sine_x.shape[0], 1))
        sine_y = sine_y.reshape((sine_y.shape[0], 1))

        errors_RBF(sine_x,sine_y)


