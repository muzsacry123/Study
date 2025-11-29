import numpy as np
import matplotlib.pyplot as plt

trainLoss = []
with open('diff_learning_rates.txt', 'r') as f:
    for line in f.readlines():
        trainLoss.append(line.split())
        
trainLoss = np.array(trainLoss).astype('float64')        
        

plt.figure()
plt.plot(trainLoss[0], label = '$\eta=10^{-1}$')
plt.plot(trainLoss[1], label = '$\eta=10^{-2}$')
plt.plot(trainLoss[2], label = '$\eta=10^{-3}$')
plt.xlabel('num of epochs')
plt.ylabel('Training average log-likelihood')
plt.legend()
plt.show()