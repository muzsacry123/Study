
import numpy as np
import math
import inspection as ins
import csv
import argparse
from collections import Counter


labels = []
features = []
attrs = []
data = []

with open('heart_train.tsv', 'r') as f:
    file = csv.reader(f, delimiter='\t')
    for line in file:               # for every single line in the file, line[] is a row vector
        if line[0].isdigit():           # if the first element of line[] is a digit
            # store data, including the label column (e.g. include 'heart disease')
            data.append(line[:])
            # store label column only
            labels.append(line[-1])
        else:                           # if the first element of line[] is an attribute
            attrs.append(line[:-1])     # here attrs is a list [[1x8]]
    data = np.array(data)
    data = data.astype(int)
    
    # for decision tree use        
    for row in data:
        attrDict = {}           # define a dictionary for attributes
        for i in range(len(attrs[0])):   
            attrDict[attrs[0][i]] = row[i]        # the i-th-column attribute in the dictionary has the value row[i] (i-th column in this row)
        
        features.append(attrDict)           # a list containing dictionary
                
print(data)