# 10-601 Homework 1 - Majority Vote Algorithm
# Aomeng Li, 01/25/2023


import numpy as np
import argparse


def readFile(fileName):
    
    label = []
    feature = []
    
    with open(fileName,'r') as f:           # close the file handler when done.
        for line in f.readlines():          # for every single line in the file, line[] is a row vector
            if line[0].isdigit():           # if the first element of line[] is a digit
                feature.append(line[:-2])   # store data, not including the last 2 columns ('heart disease' and '\n')
                label.append(line[-2])      # store the last but not least column
    
    return label, feature


def train(label, feature):  # a majority-vote training algorithm
    
    frequency = 0
    
    for item in set(label):                 # for 0's and 1's in label[]
        if label.count(item) > frequency:   # compare the occurences that 0 or 1 shows up
            frequency = label.count(item)
            prediction = item
        else:
            if label.count(item) == frequency:
                if item > prediction:
                    frequency = label.count(item)
                    prediction = item
                
    return prediction     # This is the prediction label of majority vote classifier
    

def predict(feature, prediction):
    
    results = []
    
    for line in feature:
        results.append(prediction)    # just label the feature with the prediction of majority vote classifier
        
    return results


def error(predictLabel, trueLabel):
    
    predictLabel = np.array(predictLabel)
    trueLabel = np.array(trueLabel)
    fail = np.sum(predictLabel != trueLabel)
    total = predictLabel.size
    
    return fail / total


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input',  type = str, help = 'path to the training input .tsv file')
    parser.add_argument('test_input',   type = str, help = 'path to the test input .tsv file')
    parser.add_argument('train_out',    type = str, help = 'path of output .txt file to which the predictions on the training data should be written')
    parser.add_argument('test_out',     type = str, help = 'path of output .txt file to which the predictions on the test data should be written')
    parser.add_argument('metrics_out',  type = str, help = 'path of the output .txt file to which metrics such as train and test error should be written')
    
    args = parser.parse_args()
    
    trainLabel, trainFeature = readFile(args.train_input)
    testLabel, testFeature = readFile(args.test_input)
    
    prediction = train(trainLabel, trainFeature)
    
    trainResult = predict(trainFeature, prediction)
    testResult = predict(testFeature, prediction)
    
    trainError = error(trainResult, trainLabel)
    testError = error(testResult, testLabel)
    
    
    with open(args.train_out, 'w') as f1:
        for item in trainResult:
            f1.write(item + '\n')
    with open(args.test_out, 'w') as f2:
        for item in testResult:
            f2.write(item + '\n')
    with open(args.metrics_out, 'w') as f3:
        f3.write('error(train): %.7f\n' %trainError)
        f3.write('error(test): %.7f' %testError)
    