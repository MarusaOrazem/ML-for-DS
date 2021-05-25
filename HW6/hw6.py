
import numpy as np
import pandas as pd
import random
import time
from scipy.optimize import fmin_l_bfgs_b
import scipy.stats as st
from hw_kernels import KernelizedRidgeRegression, RBF
from hw_tree import RandomForest

def identity(x):
    return x

def mse(pred,y):
    #diff = (pred-y)**2
    #return np.mean(diff)
    return np.mean((np.reshape(pred, (pred.shape[0],)) - y) ** 2)

class ANNModel():

    def __init__(self, weights, units, lambda_, layers, last, loss):
        self.weights = weights
        self.units = units
        self.lambda_ = lambda_
        self.layers = layers
        self.last = last
        self.loss = loss


    def predict(self, X):
        a = feed_forward(self.weights, X, self.layers, self.last)[-1]
        return a


class ANNRegression:

    def __init__(self, units, lambda_ ):
        self.units = units
        self.lambda_ = lambda_
        self.last = identity
        self.loss = mse

    def fit(self, X, y):
        m = X.shape[1]
        self.classes = 1
        layer_size = [m] + self.units + [1] #CHANGE THIS WHEN Y WILL HAVE ALL CLASSES

        #create matrix of weights
        #weights0 = np.random.normal(size=(layer_size[-1],layer_size[0]))
        #weight0 = np.random.normal(size=(np.prod(layer_size) + 2*(len(layer_size)-1)))
        #weights0 = [ 0.81217268, -0.30587821, -0.26408588, -0.53648431,  0.43270381, -1.15076935, 0.87240588, -0.38060345,  0.15951955, -0.12468519,  0.73105397, -1.03007035]
        #print(f'Weights {weights0}')
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

        def f(w) : return cost_function(w,X,y, layer_size, self.lambda_, self.last, self.loss)
        def df(w) : return gradient(w,X,y,layer_size, self.lambda_, self.last, self.loss)


        #print(f'############ {weights_test.shape}')
        #print(test_gradient(f,df ,layer_size, self.lambda_, self.last, self.loss))

        x0_ = [np.random.normal(loc=0, scale=0.5, size=(layer_size[i] + 1, layer_size[i + 1])) for i in range(len(layer_size) - 1)]
        x0 = []
        for j in x0_:
            x0 += list(j.flatten())
        weights0 = np.array(x0)

        weights_opt, _, _ = fmin_l_bfgs_b(cost_function, weights0, fprime=gradient, args=(X, y, layer_size, self.lambda_, self.last, self.loss), maxfun=1e6, maxiter=1e3, factr=1e9)
        return ANNModel(weights_opt, self.units, self.lambda_, layer_size, self.last, self.loss)

class ANNClassification:

    def __init__(self, units, lambda_ ):
        self.units = units
        self.lambda_ = lambda_
        self.last = softmax
        self.loss = log_loss


    def fit(self, X, y):
        m = X.shape[1]
        last_layer = len(np.unique(y))
        self.classes = last_layer
        layer_size = [m] + self.units + [last_layer] #CHANGE THIS WHEN Y WILL HAVE ALL CLASSES

        #create matrix of weights
        #weights0 = np.random.normal(size=(layer_size[-1],layer_size[0]))
        #weight0 = np.random.normal(size=(np.prod(layer_size) + 2*(len(layer_size)-1)))
        #weights0 = [ 0.81217268, -0.30587821, -0.26408588, -0.53648431,  0.43270381, -1.15076935, 0.87240588, -0.38060345,  0.15951955, -0.12468519,  0.73105397, -1.03007035]
        #print(f'Weights {weights0}')
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

        def f(w) : return cost_function(w,X,y, layer_size, self.lambda_, self.last, self.loss)
        def df(w) : return gradient(w,X,y,layer_size, self.lambda_, self.last, self.loss)


        #print(f'############ {weights_test.shape}')
        #print(test_gradient(f,df ,layer_size, self.lambda_, self.last, self.loss))
        x0_ = [np.random.normal(loc=0, scale=0.5, size=(layer_size[i] + 1, layer_size[i + 1])) for i in range(len(layer_size) - 1)]
        x0 = []
        for j in x0_:
            x0 += list(j.flatten())
        weights0 = np.array(x0)


        weights_opt, _, _ = fmin_l_bfgs_b(cost_function, weights0, fprime=gradient, args=(X, y, layer_size, self.lambda_, self.last, self.loss), maxfun=1e6, maxiter=1e3, factr=1e9)
        return ANNModel(weights_opt, self.units, self.lambda_, layer_size, self.last, self.loss)

def grid(weights, layers):
    '''
    grad_flatten = []
    for i in weights:
        grad_flatten += list(i.flatten())
    weights = grad_flatten'''
    #print(layers)
    grid = []
    for i in range(len(layers)-1):
        try:
            m, n = layers[i]+1, layers[i+1]
            w = weights[:m*n]
            grid.append(np.reshape(w,(m,n)))
            weights = weights[m*n:]
        except:
            print('##############3')
    return grid



def cost_function(weights,X,y, layers, lambda_, last, loss):
    #weights = grid(weights, layers)
    #a = np.dot(weights,X.T)
    #print(f'a : {a}')
    #pred = sigmoid(a)
    a = feed_forward(weights,X,layers, last)[-1]
    #pred = sigmoid(a)
    #last = softmax(pred) #DO WE NEED TO USE SOFTMAX AT LAST STEP?
    weights_grid = grid(weights, layers)
    norms_w = 0
    for w in weights_grid:
        norms_w += np.sum(w[:-1,]**2)
    return loss(a,y) + (lambda_/2)*norms_w

def sigmoid(x):
    #print(1 / (1+np.exp(-x)))
    x = np.array(x,dtype=float)
    return 1 / (1+np.exp(-x))

def inv_sigmoid(x):
    #print(f'invers: {np.exp(-x)/(1+np.exp(-x))**2}')
    x = np.array(x, dtype=float)
    return np.exp(-x)/(1+np.exp(-x))**2

def log_loss(pred,y):
    #vektor 0, na item mestu 1
    temp = [i for i in range(len(y))]
    y = np.array(list(map(lambda x: int(x), y)))
    losses = np.log(pred[temp,y])
    return -np.sum(losses) /len(y) #ALI JE TREBA TU DELIT Z DOLŽINO KLASOV?

def log_loss2(pred,y):
    pred = np.array(pred)
    pred[pred == 0] = 1e-4
    return -np.sum(np.multiply(y,np.log(pred)))/len(y)

def softmax(x):
    norm = np.sum(np.exp(x), axis=1, keepdims=True)
    return np.exp(x) / norm

def feed_forward(weights,X, layers, last):
    weights = grid(weights, layers)


    ones = np.array([np.ones(X.shape[0])]).T
    a=np.append(X, ones, axis=1)
    a_list = [a]
    for weight in weights[:-1]:
        #print(f'a : {a}')
        a = np.array(sigmoid(a.dot(weight)))
        ones = np.array([np.ones(a.shape[0])]).T
        a = np.append(a, ones, axis=1)
        #a = np.dot( a,weight.T)
        #a= a.dot(weight)
        a_list.append(a)


    a_list.append(last(a.dot(weights[-1])))
    return a_list


def gradient(weights,X,y, layers, lambda_, last, loss):
    a = feed_forward(weights,X, layers, last)

    y = np.array(list(map(lambda x: int(x), y)))
    deltas = []

    weights = grid(weights, layers)

    grad = [np.zeros(w.shape) for w in weights]
    delta = [[] for _ in weights]

    if loss == mse:
        delta[-1] = 2 * (a[-1] - np.reshape(y, a[-1].shape))
        grad[-1] = a[-2].T.dot(delta[-1])
    if loss == log_loss:
        delta[-1] = a[-1]
        delta[-1][range(len(y)), y] -= 1
        # delta[-1] /= len(y)
        grad[-1] = a[-2].T.dot(delta[-1])
    for i in range(1, len(weights)):
        act_i = np.delete(a[-1 - i] * (1 - a[-1 - i]), -1, axis=1)
        weights_i = np.delete(weights[-i], -1, axis=0).T
        delta[-1 - i] = act_i * delta[-i].dot(weights_i)
        grad[-1 - i] = a[-2 - i].T.dot(delta[-1 - i])

    #flat and normalize it
    grad_flatten = []
    #regularize
    reg = []
    for w in weights:
        zeros = np.zeros((1, w.shape[1]))
        reg+= list(np.append(w[:-1, ], zeros, axis = 0)*lambda_)

    #grad_reg = [lambda_ * np.append(w[:-1, ], np.zeros((1, w.shape[1])), axis=0) for w in weights]
    for j in grad:
        grad_flatten += list(j.flatten()*(1/len(y)))

    gg= []
    for j in reg:
        gg+=list(j.flatten())


    return (np.array(grad_flatten) + np.array(gg)).squeeze()


def test_gradient(f,df, layers, lambda_, last, loss):
    eps = 1e-10
    tol = 1e-3
    n = sum([layers[i]*layers[i+1]+1 for i in range(len(layers)-1)]) #+ sum([i for i in layers[:-1]]) #weights + bias
    print(n)
    for _ in range(100):
        #x0 = np.random.normal(size=n)
        x0_ = [np.random.normal(loc=0, scale=0.5, size=(layers[i] + 1, layers[i + 1])) for i in range(len(layers) - 1)]
        x0 = []
        for j in x0_:
            x0 += list(j.flatten())
        x0 = np.array(x0)

        for i in range(n):
            e_i = np.zeros(len(x0))
            e_i[i] = eps
            grad = df(x0)
            grad_flatten = []

            for j in grad:
                grad_flatten += list(j.flatten())
            grad_flatten = np.array(grad_flatten)

            fplus = f(np.add(x0 , e_i))
            fminus = f(x0)

            if (fplus - fminus) / (eps)  - grad_flatten[i] > tol:
                return False
            else:
                print('-------------')

    return True

def cv(ann, X,y, units, lambdas, loss):
    n = 80 * len(y) // 100
    k = 10
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]

    # set layers
    M = units
    best_lambdas = []
    final_rmse = []
    final_all_rmse = []
    for m in M:
        #print(f"Units: {m}")
        for l in lambdas:
            #print(f"lambda: {l}")
            reg = ann(units = m, lambda_ = l)
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

                x_test_cv = x_train[indexes_folds[i]].reshape((len(indexes_folds[i]), len(X[0])))
                y_test_cv = y_train[indexes_folds[i]]

                x_train_cv = np.array([])
                y_train_cv = np.array([])
                for j in all_folds:
                    x_train_cv = np.append(x_train_cv, x_train[indexes_folds[j]])
                    y_train_cv = np.append(y_train_cv, y_train[indexes_folds[j]])

                x_train_cv = x_train_cv.reshape(((k - 1) * n // k, len(X[0])))
                # fit model and calculate RMSE
                model = reg.fit(x_train_cv, y_train_cv)
                predicted = model.predict(x_test_cv)
                rmse = loss(predicted, y_test_cv)
                all_rmse.append(rmse)
            final_all_rmse.append(all_rmse)
            final_rmse.append([np.mean(all_rmse), m ,l])


    i = final_rmse.index(min(final_rmse, key = lambda x:x[0]))
    rmses = final_all_rmse[i]
    interval  = st.t.interval(0.95, len(rmses) - 1, loc=np.mean(rmses), scale=st.sem(rmses))
    return final_rmse[i], interval

def cv_h2(X,y, loss):
    n = 80 * len(y) // 100
    k = 10
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]

    # test polynomial kernel
    sigmas = np.arange(1, 20, 0.1)
    lambdas = np.arange(0.001, 1, 0.01)
    final_rmse = []
    final_all_rmse = []
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
                rmse = loss(y_test_cv, predicted)
                all_rmse.append(rmse)
            final_all_rmse.append(all_rmse)
            final_rmse.append([np.mean(all_rmse), sigma, l])

        i = final_rmse.index(min(final_rmse, key=lambda x: x[0]))
        rmses = final_all_rmse[i]
        interval = st.t.interval(0.95, len(rmses) - 1, loc=np.mean(rmses), scale=st.sem(rmses))
        return final_rmse[i], interval

def cv_h3(X,y, loss):
    n = 80 * len(y) // 100
    k = 10
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]


    final_rmse = []
    final_all_rmse = []
    reg = RandomForest(random.Random(1), 100, 2)
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
        model = reg.build(x_train_cv, y_train_cv)
        predicted = model.predict(x_test_cv)
        rmse = loss(y_test_cv, predicted)
        all_rmse.append(rmse)
    final_all_rmse.append(all_rmse)
    final_rmse.append([np.mean(all_rmse)])

    i = final_rmse.index(min(final_rmse, key=lambda x: x[0]))
    rmses = final_all_rmse[i]
    interval = st.t.interval(0.95, len(rmses) - 1, loc=np.mean(rmses), scale=st.sem(rmses))
    return final_rmse[i], interval

def housing2():
    df = pd.read_csv('housing2r.csv', sep=',')
    X = df.iloc[:, :-1].to_numpy()
    X = normalize(X)
    y = df.iloc[:, -1].to_numpy()

    t1 = time.time()
    c = cv(ANNRegression, X, y,
           [[],[5,2,5], [5,2,2],[20],[5,10,5],[3,3,10,3,3],[15,10]], [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1], mse)
    t2 = time.time()
    print(c)

    c_rbf = cv_h2(X,y,mse)
    print(c_rbf)

    n = 80 * len(y) // 100
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]

    m1 = ANNRegression(c[0][1], c[0][2]).fit(x_train, y_train)
    m2 = KernelizedRidgeRegression(RBF(c_rbf[0][1]), c_rbf[0][2]).fit(x_train, y_train)
    names = ['ANN', 'KernelizedRidgeRegression']
    i = 0
    for m in [m1,m2]:
        p = m.predict(x_test)
        print(names[i])
        print(mse(p,y_test))
        i+=1



def housing3():
    df = pd.read_csv('housing3.csv', sep=',')
    X = df.iloc[:, :-1].to_numpy()
    X = normalize(X)
    y = df.iloc[:, -1]
    y = np.asarray(y == "C1", dtype=int)

    n = 80 * len(y) // 100
    x_train, y_train = X[:n, :], y[:n]
    x_test, y_test = X[n:, :], y[n:]
    '''
    c = cv(ANNClassification, X, y,
           [[],[2,2,2],[5,5,5],[2,13,2],[13],[10,13,4]], [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1], log_loss)
    print(c)

    c2 = cv_h3(X,y, log_loss)
    

    m1 = ANNClassification(c[0][1], c[0][2]).fit(x_train, y_train)
    p = m1.predict(x_test)
    print('ANN')
    print(log_loss(p, y_test))'''

    m2 = RandomForest(random.Random(1),100,2).build(x_train,y_train)
    p = m2.predict(x_test)
    print('RandomForest')
    print(log_loss2(p,y_test))

def huge_dataset():
    df = pd.read_csv('train.csv', sep=',')
    X = df.iloc[:, :-1].to_numpy()
    X = normalize(X)
    y = df.iloc[:, -1].to_numpy()
    print(np.unique(y))
    y = np.array(list(map(lambda x: int(x[-1])-1, y)))
    #y = np.asarray(y == "C1", dtype=int)

    t1 = time.time()
    c = cv(ANNClassification, X, y,
           [[15,15,15,15,15],[10,10,10],[20,20,20],[10,10],[50,50,50,50],[80,80]], [0.0000001, 0.000001, 0.00001], log_loss)
    t2 = time.time()
    print(t2 - t1)
    print(c)






def normalize(X):
    up = (X - np.mean(X, axis=0))
    down = np.std(X, axis=0)
    return up / down


if __name__ == '__main__':
    np.random.seed(1)
    '''
    X = pd.read_csv('housing3.csv')[:1][['CRIM', 'INDUS']].to_numpy()
    y = pd.read_csv('housing3.csv')[:1][['Class']].to_numpy()
    y = np.asarray(y=='C1', dtype = int)
    '''
    X = np.array([[1, 0.3], [0.5, 0.6],[0.1,0.3], [0.3,0.6], [1,0], [1, 0.3], [0.5, 0.6],[0.1,0.3]])
    y = np.array([0,1,1,1,0, 0,1,1])

    #ann = ANNClassification([], 0)

    #model = ann.fit(X,y)
    #p = model.predict(X)
    #print(p)
    housing3()
    #huge_dataset()