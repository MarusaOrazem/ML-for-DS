import scipy.optimize as sc
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler


def softmax(U):
    result = []
    divider = 0
    for i in U:
        divider += math.e ** i
    for u in U:
        result.append( (math.e ** u) /divider)

    return result

def log_likelihood(beta, *args ):
    X,y = args
    beta = beta.reshape((max(y), len(X[0])))
    l = 0
    for i, x in enumerate(X):
        #print(beta)
        #print(x)
        U = np.dot(beta, x)
        U = np.append(U, [0])
        soft = softmax(U)
        predict = y[i]
        pij = soft[predict]
        l -= math.log(pij)
    return l


class MultinomialLogReg:

    def __init__(self):
        self.name = "bla"

    def build(self,X,y):
        width = len(X[0])
        height = max(y)
        beta = 0.5*np.ones((height, width))
        #print(beta)


        ret = sc.fmin_l_bfgs_b(log_likelihood, beta, args = (X,y), approx_grad = True)
        #print(ret)
        self.beta = ret[0].reshape((height,width))

        return ret

    def predict(self, X):
        predict = []
        for x in X:
            U = np.dot(self.beta, x)
            U = np.append(U, [0])
            soft = softmax(U)
            index = soft.index(max(soft))
            predict.append(index)

        return predict

def inv_logit(x):
    print(x)
    return 1./(1.+math.e**(-x))


def ordinal_log_likelihood(beta, *args):
    X, y = args
    delta, beta = beta[:len(y)-2], beta[len(y)-2:]
    t = np.cumsum(delta)
    t = np.append([-np.inf, 0], t)
    t = np.append(t,[np.inf])
    #print(f"t: {t}")
    #print(f"beta: {beta}")
    l = 0
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
        l -= math.log(pi)
    return l


class OrdinalLogReg:

    def __init__(self):
        self.name = "bla"

    def build(self, X, y):
        beta = np.ones(len(y)-2 + len(y)+1)
        #print(beta)
        #beta = np.array([ 1,  1,  1, 1,1])

        ret = sc.fmin_l_bfgs_b(ordinal_log_likelihood, beta, args=(X, y), approx_grad=True)
        #print(ret)
        self.deltas = ret[0][:len(y)-2]
        self.beta = ret[0][len(y)-2:]
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
            for i in range(len(t)):
                if t[i]>=ui:
                    predict.append(i-1)
                    break
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
    return 1-count/len(real)



if __name__ == "__main__":
    print("Hello")
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

    multinomial = MultinomialLogReg()
    multinomial.build(x_train,y_train)
    predict = multinomial.predict(x_test)
    print(missclassification(predict, y_test))






