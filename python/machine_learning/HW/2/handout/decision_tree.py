
import numpy as np
import csv
from collections import Counter
import math
import argparse


class Node:
    
    def __init__(self, attr, attrValue, left, right,*, value=None):
        
        self.attr = attr
        self.attrValue = attrValue
        self.left = left
        self.right = right
        self.value = value
    
    
    # check if is a leaf node
    def isLeaf(self):
        
        return self.value is not None



class DecisionTree:
    
    def __init__(self, maxDepth, minSplit=2, numAttrs=None):
        
        self.minSplit = minSplit
        self.maxDepth = maxDepth
        self.numAttrs = numAttrs
        self.root = None
    
    
    # Main function
    def train(self, X, y):
        
        if not self.numAttrs:       # if the number of attributes is not defined
            self.numAttrs = X.shape[1]  
        else:
            self.numAttrs = min(X.shape[1], self.numAttrs)
            
        self.root = self.growTree(X, y)
    
    
    # Main function: To grow a tree by training
    def growTree(self, X, y, depth=0):
        
        n_samples, n_attrs = X.shape
        n_labels = len(np.unique(y))     # return number of unique values e.g. for ['0' '1'] return 2
        
        # check the stopping criteria
        if depth >= self.maxDepth or n_labels == 1 or n_samples < self.minSplit:
            leaf_value = self.majorVote(y)
            return Node([],[],[],[],value = leaf_value)
        
        attrs = np.random.choice(n_attrs, self.numAttrs, replace=False)
        
        # find the best split attribute
        bestAttr, bestAttrValue = self.splitAttr(X, y, attrs)
        
        # create child nodes
        leftIdxs, rightIdxs = self.split(X[:, bestAttr], bestAttrValue)
        left = self.growTree(X[leftIdxs, :], y[leftIdxs], depth+1)
        right = self.growTree(X[rightIdxs, :], y[rightIdxs], depth+1)
        
        return Node(bestAttr, bestAttrValue, left, right)
    
    
    # To find the most common label thru majority vote algo
    def majorVote(self, y):
        
        counter = Counter(y)
        value = counter.most_common(1)[0][0]        # returns [('1', # of this label)]
        
        return value
    
    
    # Split the data according to the best attribute
    def splitAttr(self, X, y, attrs):
        
        bestMutualInfo = -1
        splitAttr, splitAttrValue = None, None      # the name of attribute to be split and its value (0 or 1)
        
        for currAttr in attrs:
            Xcol = X[:, currAttr]
            attrValues = np.unique(Xcol)
            
            for attrValue in attrValues:
                # calculate the mutual information
                mutualInfo = self.calcMutualInfo(y, Xcol, attrValue)
                
                if mutualInfo > bestMutualInfo:
                    bestMutualInfo = mutualInfo
                    splitAttr = currAttr
                    splitAttrValue = attrValue
        
        return  splitAttr, splitAttrValue
            
            
    # calculate the mutual information
    def calcMutualInfo(self, y, Xcol, attrValue):
        
        # base entropy or parent entropy
        baseEntropy = self.calcEntropy2(y)
        
        # create child
        leftIdxs, rightIdxs = self.split(Xcol, attrValue)
        
        if len(leftIdxs) == 0 or len(rightIdxs) == 0: 
            return 0
        
        # calculate conditional entropy of child
        n_samples = len(y)
        n_leftSamples, n_rightSamples = len(leftIdxs), len(rightIdxs)
        leftEntropy, rightEntropy = self.calcEntropy2(y[leftIdxs]), self.calcEntropy2(y[rightIdxs])
        
        condEntropy = (n_leftSamples / n_samples) * leftEntropy + (n_rightSamples / n_samples) * rightEntropy
        
        # calculate the MI
        mutualInfo = baseEntropy - condEntropy
        
        return mutualInfo
        
    
    # A dictionary to inspect the number of 0's and 1's (or maybe 2's or more) in the label
    def calcLabelCount(self, labels):

        labelCount = {}     # define a dictionary to store the labels and their numbers of appearance

        for value in labels:
            # if this value (0 or 1) appears for the first time
            if value not in labelCount:
                labelCount[value] = 1       # then count 1, just to initialize
            else:                           # if this value appears again
                labelCount[value] += 1      # then add the count

        return labelCount
     
    
    # calculate the entropy
    def calcEntropy(self, y):
        
        labelCount = self.calcLabelCount(y)
        
        entropy = 0     # initialize
        totalCount = sum(labelCount.values())    # total counts in labels

        for count in labelCount.values():
            # apply the entropy formula
            entropy += -(count/totalCount * math.log2(count/totalCount))

        return entropy
    
    
    # calculate the entropy, second method
    def calcEntropy2(self, y):
        
        labelCount = np.bincount(y)
        probs = labelCount / len(y)
        
        entropy = -np.sum([p * np.log2(p) for p in probs if p>0])
        
        return entropy
    
    
    # Split
    def split(self, Xcol, attrValue):
        
        leftIndex = np.argwhere(Xcol <= attrValue).flatten()
        rightIndex = np.argwhere(Xcol > attrValue).flatten()
        
        return leftIndex, rightIndex
    
    
    # predict the node
    def predict(self, X):
        
        return np.array([self.traverseTree(x, self.root) for x in X])
    
    
    # traverse a tree
    def traverseTree(self, x, node):
        
        if node.isLeaf():
            return node.value
        
        if x[node.attr] <= node.attrValue:
            return self.traverseTree(x, node.left)
        return self.traverseTree(x, node.right)
    


def loadData(fileName):

    labels = []
    attrValues = []
    attrs = []
    data = []

    # close the file handler when done.
    with open('heart_train.tsv', 'r') as f:
        file = csv.reader(f, delimiter='\t')
        for line in file:               # for every single line in the file, line[] is a row vector
            if line[0].isdigit():           # if the first element of line[] is a digit
                # store data, excluding the label column (e.g. exclude 'heart disease' labels)
                data.append(line[:-1])
                # store label column only
                labels.append(line[-1])
            else:                           # if the first element of line[] is an attribute
                attrs.append(line[:-1])     # attrs is a list [[1xcol]] with all the attribute names
                
        data = np.array(data).astype(int)
        labels = np.array(labels).astype(int)       # convert into int arrays

    # for decision tree use        
    for row in data:
        attrDict = {}           # define a dictionary for attributes
        for i in range(len(attrs[0])):   
            attrDict[attrs[0][i]] = row[i]        # the i-th-column attribute in the dictionary has the value row[i] (i-th column in this row)
        
        attrValues.append(attrDict)           # attrValues is a list [nxcol] containing dictionary attrDict
        
            
    return labels, attrValues, np.array(data), attrs

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input',  type=str,
                        help='path to the training input .tsv file')
    parser.add_argument('test_input',  type=str,
                        help='path to the test input .tsv file')
    parser.add_argument('max_depth',    type=int,
                        help='The maximal depth of the tree')
    parser.add_argument('train_out',    type = str, 
                        help = 'path of output .txt file to which the predictions on the training data should be written')
    parser.add_argument('test_out',     type = str, 
                        help = 'path of output .txt file to which the predictions on the test data should be written')
    parser.add_argument('metrics_out',  type = str, 
                        help = 'path of the output .txt file to which metrics such as train and test error should be written')

    args = parser.parse_args()
    
    y_train, _, X_train, _ = loadData(args.train_input)
    y_test, _, X_test, _ = loadData(args.test_input)
    
    tree = DecisionTree(args.max_depth)
    tree.train(X_train, y_train)
    
    yTrain_pred = tree.predict(X_train)
    trainError = 1 - np.sum(y_test == yTrain_pred) / len(y_train)
    
    with open(args.train_out, "w") as f1:
        for i in range(len(yTrain_pred)):
            f1.write("{}\n".format(yTrain_pred[i]))
    
    yTest_pred = tree.predict(X_test)
    testError = 1 - np.sum(y_test == yTest_pred) / len(y_test)

    with open(args.test_out, "w") as f2:
        for i in range(len(yTest_pred)):
            f2.write("{}\n".format(yTest_pred[i]))
        
    
    with open(args.metrics_out, "w") as f:
        f.write('error(train): %.6f\n' %trainError)
        f.write("error(test): %.6f\n" %testError)