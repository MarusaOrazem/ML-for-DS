import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_):
        self.kernel = kernel
        self.l = lambda_

    def fit(self,X,y):
        para = np.linalg.inv(self.kernel(X,X)+self.l*np.eye(np.shape(X)[0])).dot(y)
        return Model(X, para, self.kernel)

class Model:
    def __init__(self,X,param,kernel):
        self.X = X
        self.param = param
        self.kernel = kernel

    def predict(self,XX):
        return self.param.dot(self.kernel(self.X,XX))

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

def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def sine_plot():
    df = pd.read_csv('sine.csv', sep=',')
    X = df.iloc[:, :-1].to_numpy()
    X_t = X
    y = df.iloc[:, -1].to_numpy()
    X = normalize(X)

    kernels = [Polynomial(15), RBF(0.2)]
    names = ['Polynomial kernel', 'RBF kernel']
    colors = ['r', 'g']

    for i in range(2):
        k = kernels[i]
        reg = KernelizedRidgeRegression(k, 1)
        model = reg.fit(X, y)
        predictions = model.predict(X)
        list1, list2 = zip(*sorted(zip(X_t, predictions)))
        plt.plot(list1, list2, label=names[i], color=colors[i])

    plt.scatter(X_t, y, color='b', label='data  points', alpha=0.2)
    plt.legend()
    plt.show()
    plt.savefig("sine.png")

def RMSE(y1,y2):
    res = 0
    for i in range(len(y1)):
        res += (y1[i]-y2[i])**2
    return np.sqrt(res / len(y1))

def errors_polynomial(X,y):
    n = 80 * len(y) // 100
    k = 10
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]

    # test polynomial kermel
    M = [i for i in range(1, 11)]
    lambdas = np.arange(0.5, 30, 0.5)
    best_lambdas = []
    for m in M:
        final_rmse = []
        for l in lambdas:
            reg = KernelizedRidgeRegression(Polynomial(m), l)
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

        best_lambdas.append(final_rmse.index(min(final_rmse)))

    # results for constant lambda
    '''
    best_rmse_costant = []
    for m in M:
        reg = KernelizedRidgeRegression(Polynomial(m), 1)
        model = reg.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rmse = RMSE(y_test, predicted)
        best_rmse_costant.append(rmse)'''

    best_rmse_cv = []
    # results for constant lambda
    best_rmse_costant = []
    for i, m in enumerate(M):
        reg = KernelizedRidgeRegression(Polynomial(m), 1)
        model = reg.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rmse = RMSE(y_test, predicted)
        best_rmse_costant.append(rmse)

        reg1 = KernelizedRidgeRegression(Polynomial(m), lambdas[i])
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
    plt.show()

def errors_RBF(X,y):
    n = 80 * len(y) // 100
    k = 10
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]

    # test polynomial kermel
    sigmas = np.arange(0.5, 20, 1)
    lambdas = np.arange(0.001, 1, 0.01)
    best_lambdas = []
    for sigma in sigmas:
        final_rmse = []
        for l in lambdas:
            reg = KernelizedRidgeRegression(RBF(sigma), l)
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

        best_lambdas.append(final_rmse.index(min(final_rmse)))

    best_rmse_cv = []
    # results for constant lambda
    best_rmse_costant = []
    for i,sigma in enumerate(sigmas):
        reg = KernelizedRidgeRegression(RBF(sigma), 1)
        model = reg.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rmse = RMSE(y_test, predicted)
        best_rmse_costant.append(rmse)

        reg1 = KernelizedRidgeRegression(RBF(sigma), lambdas[i])
        model1 = reg1.fit(x_train, y_train)
        predicted1 = model1.predict(x_test)
        rmse1 = RMSE(y_test, predicted1)
        best_rmse_cv.append(rmse1)


    plt.plot(sigmas, best_rmse_costant, color='r', label='Constant lambda = 1')
    plt.plot(sigmas, best_rmse_cv, color='g', label='Best lambda')
    plt.title("RMSE with RBF kernel")
    plt.xlabel("sigma")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #sine_plot()

    df = pd.read_csv('housing2r.csv', sep=',')
    X = df.iloc[:, :-1].to_numpy()
    X_t = X
    X = normalize(X)
    y = df.iloc[:, -1].to_numpy()
    errors_polynomial(X,y)


