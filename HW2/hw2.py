import scipy.optimize as sc
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler


def softmax(U,beta):
    result = []
    #print(f"U : {len(U)}, beta : {len(beta)} ")
    #print(U)
    #print(beta)
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
        #beta = 0.5*np.ones((height, width))
        shape = (len(list(set(y))) - 1, X.shape[1] + 1)
        beta = np.ones(shape) / 2
        #print(beta)


        ret = sc.fmin_l_bfgs_b(log_likelihood, x0 =beta, args = (X,y), approx_grad = True)
        #print(ret)
        self.beta = ret[0].reshape(shape)

        return ret

    def predict(self, X):
        predict = []
        for x in X:
            U = np.dot(self.beta[:,1:], x)
            #U = np.append(U, [0])
            soft = softmax(U, self.beta)
            index = soft.index(max(soft))
            predict.append(index)

        return predict

def inv_logit(x):
    #return 1./(1.+math.e**(-x))
    return 1 / 2 + 1 / 2 * np.tanh(x/2)


def ordinal_log_likelihood(beta, *args):
    X, y = args
    #print(beta)
    delta, beta = beta[:len(list(set(y)))-2], beta[len(list(set(y)))-2:]
    #print(delta)
    #print(beta)
    t = np.cumsum(delta)
    t = np.append([-np.inf, 0], t)
    t = np.append(t,[np.inf])
    #print(f"t: {t}")
    #print(f"beta: {beta}")
    l = 0
    eps = 1e-10
    for i, x in enumerate(X):
        #print(f"x: {x}")
        real_class = y[i]
        ui = np.dot( beta[1:],x ) + beta[0]
        #print(f"ui: {ui}")
        a = inv_logit( t[real_class+1]-ui )
        b = inv_logit( t[real_class]-ui )
        pi = a-b
        #print(f"a: {a}")
        #print(f"b: {b}")
        #print(f"pi: {pi}")
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
        self.deltas = ret[0][:len(list(set(y)))-2]
        self.beta = ret[0][len(list(set(y)))-2:]
        #print(self.deltas)
        #print(self.beta)

        return ret

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



if __name__ == "__main__":
    print("Hello")
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[0,1,1]])
    X = X / np.sum(X, axis=0)
    y = np.array([1, 0, 2,1])

    shape = (len(list(set(y))) - 1, X.shape[1] + 1)
    #beta = np.ones(shape) / 2

    width = len(X[0])
    height = max(y)
    beta2 = 0.5 * np.ones((height, width))
    #print(f"Mine beta: {beta2}")
    #for i in X:
        #U = np.dot(beta[:,1:], i)
        #print(softmax(U, beta))

    multinomial = OrdinalLogReg()
    multinomial.build(X, y)
    #predict = multinomial.predict(X)
    '''
    beta = np.array([ 1,  1, 1, 1, 1])  # 3 classi
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = x / np.sum(x, axis=0)# 3 podatki, z 3 parametri
    y = np.array([1, 0, 2])
    #U = np.dot(beta,x[2])
    #U = np.append(U,[0])
    #print(softmax(U))
    c = OrdinalLogReg()
    #print(ordinal_log_likelihood(beta, x, y))
    c.build(x,y)
    print(c.predict(x))
    '''
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
    #print(x_test)

    multinomial = OrdinalLogReg()
    multinomial.build(x_train,y_train)
    predict = multinomial.predict(x_test)
    print(missclassification(predict, y_test))






