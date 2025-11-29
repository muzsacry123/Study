
import numpy as np
import math
import inspection as ins
import csv
import argparse
import operator


def loadData(fileName):

    labels = []
    features = []
    attrs = []
    data = []

    # close the file handler when done.
    with open('heart_train.tsv', 'r') as f:
        file = csv.reader(f, delimiter='\t')
        for line in file:               # for every single line in the file, line[] is a row vector
            if line[0].isdigit():           # if the first element of line[] is a digit
                # store data, including the label column (e.g. include 'heart disease')
                data.append(line[:])
                # store label column only
                labels.append(line[-1])
            else:                           # if the first element of line[] is an attribute
                attrs.append(line[:-1])     # attrs is a list [[1xcol]] with all the attribute names

    # for decision tree use        
    for row in data:
        attrDict = {}           # define a dictionary for attributes
        for i in range(len(attrs[0])):   
            attrDict[attrs[0][i]] = row[i]        # the i-th-column attribute in the dictionary has the value row[i] (i-th column in this row)
        
        features.append(attrDict)           # features is a list [nxcol] containing dictionary attrDict
            
    return labels, features, data, attrs


class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''


    def __init__(self, data, labels, depth, maxDepth, attrs, pastAttrs, message, left, right, feature, threshold):

        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = None
        
        
        self.data = data
        self.label = labels
        self.labelCount = self.calcLabelCount(labels)
        self.Entropy = self.calcEntropy(self.labelCount)
        
        self.vote = self.majorVote()
        
        self.depth = depth
        self.maxDepth = maxDepth
        self.attrs = attrs
        self.pastAttrs = pastAttrs
        self.splitAttr = None
        self.splitAttrValue = None
        self.child = []
        
        self.message = message


    # Identify if it's a leaf node
    def isLeaf(self):
        return self.value is not None
    
    
    
    # Majority Vote
    def majorVote(self):
        voteCount = {}
        for vote in self.label:         # self.label contains the labels in the dataset (0's and 1's)
            if vote not in voteCount.keys():
                voteCount[vote] = 0
            else:
                voteCount[vote] += 1
        
        # sort the dictionary to get the values that has the maximal appearances
        sortedVoteCount = sorted(voteCount.items(), key=operator.itemgetter(1), reverse=True)
        
        return sortedVoteCount[0][0]
    
    
    # Choose the best attribute to split, with mutual information criteria
    def split(self, attrs, labelCount, data):
        
        numAttrs = len(attrs[0])        # get the number of attributes
        baseEntropy = self.calcEntropy(labelCount)     # get the root entropy
        
        bestMutualInfo = 0          # The biggest mutual information
        bestAttrs = -1              # The attribute with the biggest mutual information
        
        for attr in range(numAttrs):
            attrValues = [example[attr] for example in data]   # get the values in the column of an attribute
            uniqueVals = set(attrValues)                    # return {'0','1'} (or maybe '2' in the future) in the current attribute
            
            CondEntropy = 0
            for value in uniqueVals:
                subData, subLabelCount = self.splitData(data, attr, value)
                attrProb = len(subData) / float(len(data))
                CondEntropy += attrProb * self.calcEntropy(subLabelCount)
            
            mutualInfo = baseEntropy - CondEntropy
            
            if mutualInfo > bestMutualInfo:
                bestMutualInfo = mutualInfo
                bestAttrs = attr
        
        return bestAttrs
                
            
    
    # Split the data into sub-datasets with respect to a certain attribute and a certain value (0 or 1)
    def splitData(self, data, attr, value):
        
        subData = []
        for feature in data:        # feature is a row vector
            # if the feature at the current attribute is exactly the value we're intested in
            if feature[attr] == value:      
                updateFeature = feature[:attr]        # take the current attr col out
                updateFeature.extend(feature[attr+1:])
                subData.append(updateFeature)
        
        subLabel = subData[:-1]
        subLabelCount = self.calcLabelCount(subLabel)   # get the labelCount at the current value of attribute
        
        return subData, subLabelCount
    
    
    # Calculate the Entropy
    def calcEntropy(self, labelCount):

        return ins.calcEntropy(labelCount.values())
    
    
    # calculate the labelCount, i.e. the numbers of appearance for each label
    def calcLabelCount(self):

        return ins.calcLabelCount(self.label)


    # Calculate conditional Entropy, 
    # where attrLabel has 2 columns, 
    # Col[0] is the current values in certain attribute e.g. A,
    # Col[1] is the label with e.g. Y
    # def calcCondEntropy(self, attrLabel):

    #     attrLabelCount = {}         # define a nested dictionary to count the number of appearance of the label
    #     rowTotal = attrLabel.shape[0]       # get the row number 
        
    #     for row in attrLabel:       # the attrLabelCount should be e.g. {A=a:{Y=y:count}}
    #         attrValue = row[0]
    #         label = row[1]
            
    #         if attrValue not in attrLabelCount.keys():  
    #             attrLabelCount[attrValue] = {}
    #         if label not in attrLabelCount[attrValue].keys():
    #             attrLabelCount[attrValue][label] = 1
    #         else:
    #             attrLabelCount[attrValue][label] += 1
        
    #     condEntropy = 0
        
    #     for attrValue in attrLabelCount.keys():
    #         attrProb = sum(attrLabelCount[attrValue].values()) / rowTotal
            
    #         specCondEntropy = self.calcEntropy(attrLabelCount[attrValue].values())      # the specific conditional entropy
            
    #         condEntropy += attrProb * specCondEntropy
        
    #     return condEntropy
    
    
    # train the data
    def train(self):
        
        if self.depth == self.maxDepth:
            return
        if len(self.attrs) == len(self.pastAttrs):
            return
        if self.Entropy == 0:
            return
        
        mutualInfo = -1
        splitAttrIndex = -1
        
        for i in range(len(self.attrs)):
            if self.attrs[i] not in self.pastAttrs:
                condEntropy = self.calcCondEntropy(self.data[:,[i,-1]])
                
                if self.Entropy - condEntropy > mutualInfo:
                    splitAttrIndex = i
                    mutualInfo = self.Entropy - condEntropy
        
        self.splitAttr = self.attrs[splitAttrIndex]
        self.splitAttrValue = list(set(self.data[:,splitAttrIndex]))
        
        for value in self.splitAttrValue:
            childData = []
            for data in self.data:
                if data[splitAttrIndex] == value:
                    childData.append(data)
            childData = np.array(childData)
            
            self.message = "{} = {}: ".format(self.splitAttr, value)
            self.child.append(Node(childData, None, self.depth+1, self.maxDepth, 
                                   self.attrs, self.pastAttrs+[self.splitAttr], self.message))
            
        for child in self.child:
            child.train()
        
        return

    
    # predict the next node
    def predict(self, attrDict):
        
        if self.splitAttr == None:
            return self.vote
        
        attrValue = attrDict[self.splitAttr]
        
        for i in range(len(self.splitAttrValue)):
            if attrValue == self.splitAttrValue[i]:
                return self.child[i].predict(attrDict)
        
        return self.vote
    
    
    # print the tree
    def print(self):
        
        print("{}{}".format("| " * self.depth + self.message, self.labelCount))
        
        for child in self.child:
            child.print()
            
        return


class DecisionTree:
    '''
    A class for decision tree
    '''
    
    def __init__(self, fileName, maxDepth):
        self.label, _, self.data, self.attrs = loadData(fileName)
        self.root = Node(self.data, self.label, 0, maxDepth, self.attrs, [], [])
        
    def train(self):
        self.root.train()
        
    def predict(self, attrDict):
        return(self.root.predict(attrDict))
    
    def print(self):
        self.root.print()
        

# Write to outputs
def writeTreeFile(fileName):
    
    with open(fileName, "w") as f:
        errorCount = 0
        labels,features, _, _ = loadData(fileName)

        for label, attrDict in zip(labels, features):
            predict = tree.predict(attrDict)
            
            if predict != label:
                errorCount += 1
            f.write("{}\n".format(predict))
    
    return errorCount, labels



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input',  type=str,
                        help='path to the training input .tsv file')
    parser.add_argument('inspect_out',  type=str,
                        help='path to the inspection output .tsv file')
    parser.add_argument('max_depth',    type=int,
                        help='The maximal depth of the tree')
    parser.add_argument('train_out',    type = str, 
                        help = 'path of output .txt file to which the predictions on the training data should be written')
    parser.add_argument('test_out',     type = str, 
                        help = 'path of output .txt file to which the predictions on the test data should be written')
    parser.add_argument('metrics_out',  type = str, 
                        help = 'path of the output .txt file to which metrics such as train and test error should be written')

    args = parser.parse_args()
    
    tree = DecisionTree(args.train_input, args.max_depth)
    tree.train()
    tree.print()
    
    errorCount, labels = writeTreeFile(args.train_out)
    trainError = errorCount / len(labels)
    
    errorCount, labels = writeTreeFile(args.test_out)
    testError = errorCount / len(labels)
    
    
    with open(args.metrics_out, "w") as f:
        f.write('error(train): %.6f\n' %trainError)
        f.write("error(train): %.6f\n" %testError)
