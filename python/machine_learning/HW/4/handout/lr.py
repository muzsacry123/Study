import numpy as np
import argparse
import matplotlib.pyplot as plt


def sigmoid(x : np.ndarray):
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


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    
    # TODO: Implement `train` using vectorization
    for times in range(num_epoch):
        for i in range(X.shape[0]):
            grad = (sigmoid(np.dot(theta, X[i])) - y[i]) * X[i]     # grad(J_i), i.e. SGD algo
            theta -= learning_rate * grad
            
        # Compute negative log-likelihood over num_epoch:
        global logLikeTrain, logLikeVal
        tempTrain, tempVal = 0, 0
        
        for i in range(len(XTrain)):
            pTrain = sigmoid(theta @ XTrain[i].T)
            tempTrain += -1/len(XTrain) * (yTrain[i] * np.log(pTrain) + (1-yTrain[i]) * np.log(1-pTrain))
        logLikeTrain.append(tempTrain)
        
        for i in range(len(XVal)):
            pVal = sigmoid(theta @ XVal[i].T)
            tempVal += -1/len(XVal) * (yVal[i] * np.log(pVal) + (1-yVal[i]) * np.log(1-pVal))
        logLikeVal.append(tempVal)
        


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    
    # TODO: Implement `predict` using vectorization
    alpha = 0.5
    prediction = sigmoid(theta @ X.T).reshape(-1,1)
    prediction = np.where(prediction >= 0.5, 1, 0)
    
    return prediction.flatten()


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    
    # TODO: Implement `compute_error` using vectorization
    error = np.sum(y_pred != y) / len(y)
    
    return error


def data2vec(fileName):
    
    data = []
    with open(fileName, 'r') as f:
        for line in f.readlines():      # line = (label, [vector])
            data.append(line.split())
    data = np.array(data)
    
    y = data[:,0].astype('float64')
    X = data[:,1:].astype('float64')
    X = np.concatenate((np.ones((len(X),1)), X), axis=1)        # ???
    
    return X, y 


def writeFile(fileName, yPred):
    
    with open(fileName, 'w') as f:
        for pred in yPred:
            f.write(str(pred) + '\n')


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of gradient descent to run')
    parser.add_argument("learning_rate", type=float, 
                        help='learning rate for gradient descent')
    args = parser.parse_args()

    logLikeTrain, logLikeVal = [],[]
    
    # Read files to vectors
    XTrain, yTrain = data2vec(args.train_input)
    XVal, yVal = data2vec(args.validation_input)
    XTest, yTest = data2vec(args.test_input)
    
    # Train the data
    theta = np.zeros((XTrain.shape[1],))
    train(theta, XTrain, yTrain, args.num_epoch, args.learning_rate)
    yPredTrain = predict(theta, XTrain)
    errTrain = compute_error(yPredTrain, yTrain)
    writeFile(args.train_out, yPredTrain)
    
    # Test the data
    yPredTest = predict(theta, XTest)       # use the trained theta
    errTest = compute_error(yPredTest, yTest)
    writeFile(args.test_out, yPredTest)
    print(yTest)
    print(yPredTest)
    
    # Write the metrics file
    with open(args.metrics_out, 'w') as f:
        f.write('error(train): %.6f\n'%errTrain)
        f.write('error(test): %.6f\n'%errTest)
        
    # plot the figure
    plt.figure()
    plt.plot(logLikeTrain, label = 'training data')
    plt.plot(logLikeVal, label = 'validation data')
    plt.xlabel('# of epochs')
    plt.ylabel('Average negative log-likelihood')
    plt.legend()
    plt.show()
    
    # write file for question 7.4
    with open('diff_learning_rates.txt', 'a') as f:
        for item in logLikeTrain:
            f.write(str(item) + '\t')
        f.write('\n')
    