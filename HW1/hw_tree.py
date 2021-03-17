import numpy as np
import pandas as pd
import collections
import random
import matplotlib.pyplot as plt


def gini_index(Y):
    '''Calculates gini index'''
    n = len(Y)
    counts = collections.Counter(Y)
    gini = 1
    for key in counts.keys():
        gini -= ( counts[key] / n ) **2
    return gini


def split_dataset( data, attribute, value ):
    '''splits the data at value for attribute. return indexes in dataset for left and right node'''
    left =  np.where( data[:,attribute] <= value ) 
    right =  np.where( data[:,attribute] > value ) 
    return left, right

def get_all_splits( X, Y, fun, rand ):
    '''calculates all gini indexes for every possible split made'''
    att = []
    value = []
    qualities = []
    #select random sqrt(n) columns to be taken into account
    columns = fun( X, rand )
    if(len(columns)== 0):
        print("NO COLUMNS")
    for i in range( len( X[0] )-1 ):
        #check if attribute i is taken into account
        if( i not in columns ):
            continue

        values = np.unique(np.sort( X[ :, i ] ))
        #iterate through all unique values and split between them
        for j in range( len(values) - 1 ):
            val = (values[j] + values[j+1]) / 2
            left, right = split_dataset( X, i, val )
            left_g = gini_index( Y[left] )
            right_g = gini_index( Y[right] )
            #calculates the weightes quality of this split
            quality = len(left[0]) * left_g + len(right[0]) * right_g

            att.append( i )
            value.append( val )
            qualities.append( quality )

    return {'att': att, 'value' : value, 'qualities' : qualities}


class Node:
    def __init__(self, left, right, attribute, value):
        self.left = left
        self.right = right
        self.attribute = attribute
        self.prediction = None
        self.value = value
        
    
    def predict( self, X ):
        y = []
        for x in X:
            result = self.predict2(x)
            y.append( result )
        return y
    
    def predict2( self, x ):
        val = x[ self.attribute ]
        if( self.prediction != None ):
            return self.prediction
        elif( val <= self.value ):
            return self.left.predict2( x )
        else:
            return self.right.predict2( x )


class Tree:
    def __init__(self, rand, get_candidate_columns, min_samples):
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples
        self.majority = None
    
    
    def build( self, X, Y ):
        if( len( np.unique( Y ) ) == 1 ):
            #if the sample is homogeneous we stop spliting
            node = Node(None,None,None,None)
            node.prediction = Y[0]
            return node
        elif( len(Y) <= self.min_samples ):
            #number of instances is lower than prescribed
            #the prediction will be the majority of classes in this subset
            p = collections.Counter( Y ).most_common(1)[0][0]
            node = Node(None,None,None,None)
            node.prediction = p
            return node
        else:
            splits = get_all_splits( X, Y, self.get_candidate_columns, self.rand )
            q = splits[ 'qualities' ]
            if( len(q) == 0 ):
                #no value to split
                p = collections.Counter( Y ).most_common(1)[0][0]
                node = Node(None,None,None,None)
                node.prediction = p
                return node
            index = q.index( min( q ) )
            att = splits[ 'att' ][ index ]
            value = splits[ 'value' ][ index ]
            l, r = split_dataset( X, att, value )

            return Node( self.build( X[l], Y[l] ), self.build( X[r], Y[r] ), att, value )
    
class Bagging_trees:
    def __init__( self, trees ):
        self.trees = trees
    
    def predict( self, X ):
        results = []
        if( len( self.trees ) == 0 ):
            #"No trees to predict!"
            return ""
        #go through all the trees and predict on every one of them
        for tree in self.trees:
            results.append( tree.predict( X ) )
        
        #now look at the majority vote and take that value as prediction
        final_predict = []
        for i in range( len(results[0])  ):
            values = np.array(results)[ :, i ]
            predict = collections.Counter( values ).most_common(1)[0][0]
            final_predict.append( predict )

        return final_predict  


class Bagging:
    def __init__( self, rand, tree_builder, n ):
        self.rand = rand
        self.tree_builder = tree_builder
        self.n = n

    def build( self, X, Y ):
        #need to build self.n different trees and save them, each build on different samples of data
        trees = []
        for _ in range( self.n ):
            #build new tree
            t = Tree( self.rand, self.tree_builder.get_candidate_columns, self.tree_builder.min_samples )
            #take random instances from the data (same size as data, with replacement) and build tree from it
            #np.random.seed( self.rand )

            sample = self.rand.choices( [i for i in range(len(X))], k=len(X) )
            p = t.build( X[sample], Y[sample] )
            trees.append( p )
        return Bagging_trees( trees )
          

class RandomForest:
    def __init__( self, rand, n, min_samples ):
        self.rand = rand
        self.n = n
        self.min_samples = min_samples
    
    def build( self, X, Y ):
        t = Tree( self.rand, random_features, self.min_samples )
        b = Bagging( self.rand, t, self.n )
        self.b = b
        return b.build( X, Y )
    
    def predict( self, X ):
        return self.b.predict( X )
    

def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]
def all_features(X, rand):
    return np.arange(X.shape[1])
def random_features(X, rand ):
    return rand.choices(list(range(X.shape[1])),k=int(np.sqrt(len(X[0]))))

def missclassification( predicted, real ):
    if( len(predicted) != len(real) ):
        return ""
    if( len(real) == 0):
        return 1
    count = 0
    for i,j in enumerate( predicted ):
        if( j == real[ i ]):
            count += 1
    return 1-count/len(real)


def hw_tree_full( train, test, min_samples = 2 ):
    t = Tree( random.Random(0), all_features, min_samples )
    p = t.build(train[0], train[1])

    predict_train = p.predict(train[0])
    miss_train = missclassification( predict_train, train[1] )

    predict_test=p.predict(test[0])
    miss_test = missclassification( predict_test, test[1] )
     
    return miss_train, miss_test
    
def hw_bagging( train, test, n=50, rand = random.Random(0) ):
    t = Tree( rand, all_features, 2 )
    b = Bagging( rand, t, n )
    p = b.build( train[0], train[1] )

    predict_train = p.predict(train[0])
    miss_train = missclassification( predict_train, train[1] )

    predict_test = p.predict(test[0])
    miss_test = missclassification( predict_test, test[1] )

    return miss_train, miss_test

def hw_randomforests( train, test, n=50, rand = random.Random(0) ):
    rf = RandomForest( rand, n, 2 )
    b = rf.build(train[0], train[1])

    predict_train = b.predict(train[0])
    miss_train = missclassification( predict_train, train[1] )

    predict_test = b.predict(test[0])
    miss_test = missclassification( predict_test, test[1] )

    return miss_train, miss_test

def hw_cv_min_samples( train, test, rand = random.Random(1) ):
    #first we shuffle the data at random
    indexes = [i for i in range( len(train[0]) )]
    #np.random.seed(1)
    #np.random.shuffle( indexes )
    rand.shuffle( indexes )
    n = len(train[0])
    folds = []
    #split the data (indexes) into 5-folds
    for i in range(5):
        j = indexes[ (i)*n//5 : n*(i+1)//5 ]
        folds.append(j) #save only indexes for each fold
    #now we check the CV result for different min_samples
    candidates=[i for i in range(1,51)]
    errors = []
    er_tt = []
    for i in candidates:
        #evaluate errors for every 5 folds
        er = []
        er_t = []
        for j in range( len(folds) ):
            temp = indexes[:(j)*n//5] + indexes[(j+1)*n//5:]
            train_k_x = train[0][temp]
            train_k_y = train[1][temp]
            test_k_x, test_k_y = train[0][indexes[(j)*n//5 : n*(j+1)//5]], train[1][indexes[(j)*n//5 : n*(j+1)//5]]
            er_train, er_test = hw_tree_full( (train_k_x, train_k_y), ( test_k_x, test_k_y) , i )
            er.append(er_test)
            er_t.append(er_train)
            
        errors.append( np.sum(er)/5 )
        er_tt.append(np.sum(er_t)/5)


    plt.plot(candidates,errors)
    plt.plot(candidates,er_tt)
    plt.title("Misclassification rates vs min_samples (CV)")
    plt.xlabel("min samples")
    plt.ylabel("Misclassification rate")
    #annotate the min value of mis. rate
    min_mis = errors.index(min(errors))+1
    print(f"Minimum misclassification rate on test set: {min(errors)}")
    e_train, e_test = hw_tree_full( train, test, min_mis )
    print(f"Misclassification on whole set for min_samples = {min_mis} is:")
    print(f"    train set: {e_train}")
    print(f"    test set: {e_test}")
    plt.vlines(min_mis, 0,0.25, color='black')

    plt.savefig("miss_cv.png")
    plt.show()
    index = errors.index( min( errors ) )
    best_min_samples = candidates[ index ]
    er_train, er_test = hw_tree_full( train, test, best_min_samples )

        
    return ( er_train, er_test, best_min_samples )


if __name__ == "__main__":
    data = pd.read_csv('housing3.csv').to_numpy()
    n = int(( len(data)*80 ) / 100)
    train = data[:n,:]
    train_x, train_y = train[:, 0:(len(train[0])-1)], train[:, -1]
    test = data[n:,:]
    test_x, test_y = test[:, 0:(len(test[0])-1)], test[:, -1]

    print("1. TREE")
    e_train1, e_test1 = hw_tree_full( (train_x,train_y), ( test_x,test_y) )
    print(f"Misclassification rates:")
    print(f"    train set: {e_train1}")
    print(f"    test set: {e_test1}")

    print("2. CV")
    hw_cv_min_samples((train_x,train_y), ( test_x,test_y))
    print("3. BAGGING")
    e_train3, e_test3 = hw_bagging((train_x,train_y), ( test_x,test_y) )
    print(f"Misclassification rates:")
    print(f"    train set: {e_train3}")
    print(f"    test set: {e_test3}")

    n_trees = [1] + [i*5 for i in range(1,20)]
    rands = [0,1,2]
    colors = ['b','g','r','c','m','y','k','w']

    for r in rands:
        miss_train = []
        miss_test = []
        for i in n_trees:
            train_e, test_e = hw_bagging((train_x,train_y), ( test_x,test_y), n = i, rand = random.Random(r))
            miss_train.append( train_e )
            miss_test.append( test_e )
        plt.plot(n_trees, miss_test, label= f"seed: {r}", color = colors[r] )

    plt.legend()
    plt.title("Misclassification rate vs the number of trees (bagging)")
    plt.xlabel("Number of trees")
    plt.ylabel("Misclassification rate")
    plt.savefig("miss_b.png")

    plt.show()

    print("4. RANDOM FOREST")
    e_train4, e_test4 = hw_randomforests((train_x,train_y), ( test_x,test_y), rand = random.Random(2) )
    print(f"Misclassification rates:")
    print(f"    train set: {e_train4}")
    print(f"    test set: {e_test4}")
    for r in rands:
        print(f"rand = {r}")
        miss_train = []
        miss_test = []
        for i in n_trees:
            #print(f"tree = {i}")
            train_e, test_e = hw_randomforests((train_x,train_y), ( test_x,test_y), n = i, rand = random.Random(r))
            miss_train.append( train_e )
            miss_test.append( test_e )
        plt.plot(n_trees, miss_test, label= f"seed: {r}", color = colors[r] )
        #plt.plot(n_trees, miss_test, color = colors[r], linestyle = 'dashed')
        #patches.append( mpatches.Patch(color=colors[r], label=f'seed: {r}') ) 


    plt.legend()
    plt.title("Misclassification rate vs the number of trees (random forest)")
    plt.xlabel("Number of trees")
    plt.ylabel("Misclassification rate")
    plt.savefig("miss_rf.png")
    plt.show()


