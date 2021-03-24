import scipy.optimize as sc
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt



def softmax(U,beta):
    result = []

    divider = 0
    for k, i in enumerate(U):
        divider += math.e ** (i + beta[k][0])
    for k, u in enumerate(U):
        result.append((math.e ** (u + beta[k][0])) / (1+divider))
    result.append(1/(1+divider))

    return result

def log_likelihood(beta, *args ):
    X,y = args
    shape = (len(list(set(y))) - 1, X.shape[1] + 1)
    #beta = beta.reshape((max(y), len(X[0])))
    beta = beta.reshape(shape)
    l = 0
    for i, x in enumerate(X):
        #print(beta)
        #print(x)
        U = np.dot(beta[:,1:], x)
        #U = np.append(U, [0])
        soft = softmax(U,beta)
        predict = y[i]
        #print(f"predict: {predict}")
        pij = soft[predict]
        l -= math.log(pij)
    return l


class MultinomialLogReg:

    def __init__(self):
        self.name = "bla"

    def build(self,X,y):
        width = len(X[0])
        height = max(y)
        shape = (len(list(set(y))) - 1, X.shape[1] + 1)
        beta = np.ones(shape) / 2
        #print(beta)

        ret = sc.fmin_l_bfgs_b(log_likelihood, x0 =beta, args = (X,y), approx_grad = True)
        beta = ret[0].reshape(shape)

        model = MultinomialModel(beta)

        return model

class MultinomialModel:

    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        predict = []
        for x in X:
            U = np.dot(self.beta[:,1:], x)
            #U = np.append(U, [0])
            soft = softmax(U, self.beta)
            index = soft.index(max(soft))
            predict.append(index)

        return predict

    def log_loss(self, X,y):
        loss = 0
        for i,x in enumerate(X):
            U = np.dot(self.beta[:, 1:], x)
            real = y[i]
            soft = softmax(U, self.beta)
            l = soft[real]
            loss -= math.log(l)
        loss = loss / len(y)
        return loss


def inv_logit(x):
    return 1./(1.+math.e**(-x))

def ordinal_log_likelihood(beta, *args):
    X, y = args
    #print(beta)
    delta, beta = beta[:len(list(set(y)))-2], beta[len(list(set(y)))-2:]
    t = np.cumsum(delta)
    t = np.append([-np.inf, 0], t)
    t = np.append(t,[np.inf])
    l = 0
    eps = 1e-10
    for i, x in enumerate(X):
        real_class = y[i]
        ui = np.dot( beta[1:],x ) + beta[0]
        a = inv_logit( t[real_class+1]-ui )
        b = inv_logit( t[real_class]-ui )
        pi = a-b
        if pi < eps:
            pi = eps
        l -= math.log(pi)
    return l


class OrdinalLogReg:

    def __init__(self):
        self.name = "bla"

    def build(self, X, y):
        beta = np.ones(len(list(set(y)))-2 + len(X[0])+1)/15
        #print(beta)
        #beta = np.array([ 1,  1,  1, 1,1])
        delta0 = np.ones(len(list(set(y))) - 2) * 1e-10
        beta0 = np.ones(len(X[0]) + 1) / 15
        beta = np.append(delta0, beta0)

        ret = sc.fmin_l_bfgs_b(ordinal_log_likelihood, beta, args=(X, y), approx_grad=True)
        #print(ret)
        deltas = ret[0][:len(list(set(y)))-2]
        beta = ret[0][len(list(set(y)))-2:]
        #print(self.deltas)
        #print(self.beta)
        model = OrdinalModel( deltas, beta )

        return model


class OrdinalModel:
    def __init__(self, deltas, beta):
        self.deltas = deltas
        self.beta = beta

    def predict(self, X):
        predict = []

        t = np.cumsum(self.deltas)
        t = np.append([-np.inf, 0], t)
        t = np.append(t, [np.inf])

        for x in X:
            ui = np.dot(self.beta[1:], x) + self.beta[0]
            pi = []
            for j in range(1, len(t)):
                pi.append(inv_logit(t[j] - ui) - inv_logit(t[j - 1] - ui))
            predict.append(pi.index(max(pi)))
        return predict

    def log_loss(self, X, y):
        loss = 0

        t = np.cumsum(self.deltas)
        t = np.append([-np.inf, 0], t)
        t = np.append(t, [np.inf])

        for i,x in enumerate(X):
            ui = np.dot(self.beta[1:], x) + self.beta[0]
            pi = []
            for j in range(1, len(t)):
                pi.append(inv_logit(t[j] - ui) - inv_logit(t[j - 1] - ui))
            real = y[i]
            l = pi[real]
            loss -= math.log(l)
        loss = loss / len(y)
        return loss

def naive_log_loss(y):
    p = [0.15, 0.1, 0.05, 0.4, 0.3]
    l = 0
    for i in y:
        l -= math.log(p[i])
    loss = l/len(y)
    return loss



def missclassification( predicted, real ):
    if( len(predicted) != len(real) ):
        return ""
    if( len(real) == 0):
        return 1
    count = 0
    for i,j in enumerate( predicted ):
        if( j == real[ i ]):
            count += 1
    print(f"{count}/{len(real)}")
    return count/len(real)


def CV_log_loss(X,y,k):
    n = len(y)
    multi_losses = []
    ordinal_losses = []
    naive_losses = []

    #create indexes and shuffle them to get elements for k folds
    i_shuffled = [i for i in range(n)]
    random.seed(0)
    random.shuffle( i_shuffled )
    indexes_folds = [ ]
    for i in range(k):
        j = i_shuffled[(i) * n // k: n * (i + 1) // k]
        indexes_folds.append(j)  # save only indexes for each fold

    for i in range(k):
        all_folds = [j for j in range(k)]
        all_folds.remove(i) #remove the index for test fold

        x_test = X[ indexes_folds[i] ].reshape((len(indexes_folds[i]),len(X[0])))
        y_test = y[ indexes_folds[i] ]

        x_train = np.array([])
        y_train = np.array([])
        for j in all_folds:
            x_train = np.append( x_train, X[ indexes_folds[j] ] )
            y_train = np.append( y_train, y[ indexes_folds[j] ] )

        x_train = x_train.reshape(((k-1)*n//k,len(X[0])))
        #make all 3 models and calculate log_loss
        multi = MultinomialLogReg()
        multi_model = multi.build( x_train, y_train )
        multi_log_loss = multi_model.log_loss( x_test, y_test )
        multi_losses.append( multi_log_loss )

        ordinal = OrdinalLogReg()
        ordinal_model = ordinal.build( x_train, y_train )
        ordinal_log_loss = ordinal_model.log_loss( x_test, y_test )
        ordinal_losses.append( ordinal_log_loss )

        naive_loss = naive_log_loss(y_test)
        naive_losses.append( naive_loss )


    losses = [multi_losses, ordinal_losses, naive_losses]
    names= ['Multi', 'Ordinal', 'Naive']
    intervals = []
    for i,l in enumerate(losses):
        a = st.t.interval(0.95, len(l) - 1, loc=np.mean(l), scale=st.sem(l))
        print(f"{names[i]} : {a}")
        intervals.append( a )

    return intervals

def CV_coefficients(X,y,k):
    n = len(y)
    coefs = [[] for _ in range(12)]
    names = []
    np.random.seed(2)
    for _ in range(1000):
        indexes = np.random.choice([i for i in range(250)], 250)
        data_x = X[indexes].reshape((len(indexes),len(X[0])))
        data_y = y[indexes]
        ordinal = OrdinalLogReg()
        ordinal_model = ordinal.build(data_x, data_y)
        for i,j in enumerate(ordinal_model.beta):
            coefs[i].append(j)

    names = ['bias','age','sex','year','x1','x2','x3','x4','x5','x6','x7','x8']
    means = []
    lower = []
    upper = []
    for i, c in enumerate(coefs):
        m = np.mean(c)
        means.append(m)
        interval = st.t.interval(0.95, len(c) - 1, loc=np.mean(c), scale=st.sem(c))
        lower.append(m-interval[0] )
        upper.append(interval[1]-m )
    plt.bar(names, means, yerr =[lower,upper])
    plt.savefig("coefficients.png")

    plt.show()

    #st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))


    return 0



def create_dataset():

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    random.seed(3)

    x1 = lambda x,y : x**2 + random.uniform(-1,1)
    x2 = lambda x,y : np.sin(x)+ 2.14/2 + 2*y + random.uniform(-1,1)
    x3 = lambda x,y : -3 + random.uniform(-1,1)

    xs = [x1,x2,x3]

    for i in range(6):

        y = i
        xi = []
        for x in xs:
            xi.append(x(y,y))

        X_train.append(xi)
        y_train.append(y)

    for _ in range(1000):
        y = random.randint(0,5)
        xi = []
        for x in xs:
            xi.append(x(y,y))

        X_test.append(xi)
        y_test.append(y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    df_train = pd.DataFrame({'Class': y_train, 'X1': X_train[:,0], 'X2': X_train[:,1], 'X3': X_train[:,2]})
    df_train.to_csv('multinomial_bad_ordinal_good_train.csv',index=False,sep=';')

    df_test = pd.DataFrame({'Class': y_test, 'X1': X_test[:, 0], 'X2': X_test[:, 1], 'X3': X_test[:, 2]})
    df_test.to_csv('multinomial_bad_ordinal_good_test.csv', index=False,sep=';')

    s = StandardScaler()
    s.fit(X_train)
    X_train = s.transform(X_train)
    X_test = s.transform(X_test)

    multinomial = MultinomialLogReg()
    model = multinomial.build(X_train, y_train)
    predict = model.predict(X_test)
    print(missclassification(predict, y_test))
    loss = model.log_loss(X_test, y_test)
    print(loss)

    multinomial = OrdinalLogReg()
    model = multinomial.build(X_train, y_train)
    predict = model.predict(X_test)
    print(missclassification(predict, y_test))
    loss = model.log_loss(X_test, y_test)
    print(loss)



if __name__ == "__main__":

    df = pd.read_csv('dataset.csv', sep=";")

    #process y values
    y = df['response']
    values = ['very poor', 'poor', 'average', 'good', 'very good']
    for i, val in enumerate(values):
        a = y.where(y == val) == val
        y[a] = i
    y = y.to_numpy()

    #process sex values
    values_sex = ['M', 'F']
    for i, val in enumerate(values_sex):
        a = df['sex'].where(df['sex'] == val) == val
        df['sex'][a] = i

    df = df.drop(columns = ['response'])
    X = df.to_numpy()

    #scale data
    s = StandardScaler()
    s.fit(X)
    X = s.transform(X)

    n = len(y)
    split = 80*n // 100
    x_train, y_train = X[:split,:], y[:split]
    x_test, y_test = X[split:,:], y[split:]


    multinomial = MultinomialLogReg()
    model = multinomial.build(x_train,y_train)
    predict = model.predict(x_test)
    print(missclassification(predict, y_test))
    loss = model.log_loss(x_test,y_test)
    print(loss)

    multinomial = OrdinalLogReg()
    model = multinomial.build(x_train, y_train)
    predict = model.predict(x_test)
    print(missclassification(predict, y_test))
    loss = model.log_loss(x_test, y_test)
    print(loss)


    CV_log_loss(X,y,10)

    create_dataset()
    
    CV_coefficients(X,y,5)







