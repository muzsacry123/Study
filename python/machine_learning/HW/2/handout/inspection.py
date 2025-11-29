
import numpy as np
import math
import argparse


def loadData(fileName):

    label = []
    feature = []

    # close the file handler when done.
    with open(fileName, 'r') as f:
        for line in f.readlines():          # for every single line in the file, line[] is a row vector
            if line[0].isdigit():           # if the first element of line[] is a digit
                # store data, not including the last 2 columns (e.g. 'heart disease' and '\n')
                feature.append(line[:-2])
                # store the last but not least column
                label.append(line[-2])

    return label, feature


# A dictionary to inspect the number of 0's and 1's (or maybe 2's or more) in the label
def calcLabelCount(label):

    labelCount = {}     # define a dictionary to store the labels and their numbers of appearance

    for value in label:
        # if this value (0 or 1) appears for the first time
        if value not in labelCount:
            labelCount[value] = 1       # then count 1, just to initialize
        else:                           # if this value appears again
            labelCount[value] += 1      # then add the count

    return labelCount
    

# Calculate the label entropy at the root, take the dictionary labelCount as input
def calcEntropy(values):

    entropy = 0     # initialize
    totalCount = sum(values)    # total counts in labels

    for count in values:
        # apply the entropy formula
        entropy += -(count/totalCount * math.log2(count/totalCount))

    return entropy


# Calculate the error rate of classifying using a majority vote
def calcMajorityVoteError(values):

    maxCount = max(values)
    totalCount = sum(values)

    error = 1 - maxCount / totalCount

    return error


# calculate the entropy and error using above three functions
def dataInspect(labelCount):

    entropy = calcEntropy(labelCount.values())
    error = calcMajorityVoteError(labelCount.values())

    return entropy, error


# main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('train_input',  type=str,
                        help='path to the training input .tsv file')
    parser.add_argument('inspect_out',  type=str,
                        help='path to the inspection output .tsv file')

    args = parser.parse_args()

    label, _ = loadData(args.train_input)
    labelCount = calcLabelCount(label)

    entropy, error = dataInspect(labelCount)

    with open(args.inspect_out, 'w') as f:
        f.write('entropy: %.6f\n' % entropy)
        f.write('error: %.6f' % error)
