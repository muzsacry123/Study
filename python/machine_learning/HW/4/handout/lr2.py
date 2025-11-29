import numpy as np
import argparse
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def loss():
    global lossTrain, lossVal
    tmpTrain = 0
    tmpVal = 0
    for i in range(len(XTrain)):
        tmpTrain += (-1/len(XTrain) * (theta @ XTrain[i].T * yTrain[i]
                                       - np.log(1 + np.exp(theta @ XTrain[i].T))))
    for i in range(len(XVal)):
        tmpVal += (-1/len(XVal) * (theta @ XVal[i].T * yVal[i]
                                       - np.log(1 + np.exp(theta @ XVal[i].T))))
    lossTrain.append(tmpTrain)
    lossVal.append(tmpVal)

def train(theta, X, y, num_epoch, learning_rate):
    for i in range(num_epoch):
        print('finish %.2f%%'%((i+1)/num_epoch*100))
        for i in range(len(X)):  
            gradient = (sigmoid(theta @ X[i].T) - y[i]) * X[i]
            theta -= learning_rate * gradient
        loss()


def predict(theta, X):
    result = sigmoid(theta@X.T).T
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    return result


def compute_error(y_pred, y):
    error = np.sum(y_pred != y)
    errorRate = error / len(y)
    return errorRate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hw4')
    parser.add_argument('--formatted_train_input', type = str,
                        required=False, default = 'formatted_train_large.tsv',          
                        help = 'path to the formatted training input .tsv file')
    parser.add_argument('--formatted_validation_input', type = str,
                        required = False, default = 'formatted_val_large.tsv', 
                        help = 'path to the formatted validation input .tsv file')
    parser.add_argument('--formatted_test_input', type = str,
                        required = False, default = 'formatted_test_large.tsv',
                        help = 'path to the formatted test input .tsv file')
    parser.add_argument('--train_out', type = str,
                        required = False, default = 'train_out.txt',
                        help = 'path to output training prediction .txt file')   
    parser.add_argument('--test_out', type = str,
                        required = False, default = 'test_out.txt',
                        help = 'path to output test prediction .txt file')
    parser.add_argument('--metrics_out', type = str, 
                        required = False, default = 'metrics_out.txt',
                        help = 'path to output metrics .txt file')
    parser.add_argument('--num_epoch', type = int, 
                        required = False, default = 1000,
                        help = 'integer specifying the number of times SGD loops')
    parser.add_argument('--learning_rate', type = float, 
                        required = False, default = 0.00001,
                        help = 'float specifying the learning rate')
    
    args = parser.parse_args()
    
    lossTrain = []
    lossVal = []
    
    dataValFile = args.formatted_validation_input
    dataVal = []
    with open(dataValFile, 'r') as f:
        for line in f.readlines():
            dataVal.append(line.split())
    dataVal = np.array(dataVal)
    yVal = dataVal[:, 0].astype('float64')
    XVal = dataVal[:, 1:].astype('float64')
    XVal = np.concatenate((np.ones((len(XVal),1)), XVal), axis = 1)
    
    dataTrainFile = args.formatted_train_input
    dataTrain = []
    with open(dataTrainFile, 'r') as f:
        for line in f.readlines():
            dataTrain.append(line.split())
    dataTrain = np.array(dataTrain)
    yTrain = dataTrain[:, 0].astype('float64')
    XTrain = dataTrain[:, 1:].astype('float64')
    XTrain = np.concatenate((np.ones((len(XTrain),1)), XTrain), axis = 1)
    theta = np.zeros((XTrain.shape[1], ))
    train(theta, XTrain, yTrain, args.num_epoch, args.learning_rate)
    yPredTrain = predict(theta, XTrain)
    errorTrain = compute_error(yPredTrain, yTrain)
    with open(args.train_out, 'w') as f:
        for pred in yPredTrain:
            f.write(str(pred) + '\n')
            
    
    
    dataTestFile = args.formatted_test_input
    dataTest = []
    with open(dataTestFile, 'r') as f:
        for line in f.readlines():
            dataTest.append(line.split())
    dataTest = np.array(dataTest)
    yTest = dataTest[:, 0].astype('float64')
    XTest = dataTest[:, 1:].astype('float64')
    XTest = np.concatenate((np.ones((len(XTest),1)), XTest), axis = 1)
    yPredTest = predict(theta, XTest)
    errorTest = compute_error(yPredTest, yTest)
    with open(args.test_out, 'w') as f:
        for pred in yPredTest:
            f.write(str(pred) + '\n')

    with open(args.metrics_out, 'w') as f:
        f.write('error(train): %.6f\n'%errorTrain)
        f.write('error(test): %.6f\n'%errorTest)
    
    plt.figure()
    plt.plot(lossTrain, label = 'Train loss')
    plt.plot(lossVal, label = 'Validation loss')
    plt.xlabel('num of epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    with open('lossTrain.txt', 'a') as f:
        for item in lossTrain:
            f.write(str(item) + '\t')
        f.write('\n')