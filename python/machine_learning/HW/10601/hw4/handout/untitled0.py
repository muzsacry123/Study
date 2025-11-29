import numpy as np
import matplotlib.pyplot as plt

trainLoss = []
with open('lossTrain.txt', 'r') as f:
    for line in f.readlines():
        trainLoss.append(line.split())
        
trainLoss = np.array(trainLoss).astype('float64')        
        

plt.figure()
plt.plot(trainLoss[0], label = 'learning rate 0.001')
plt.plot(trainLoss[1], label = 'learning rate 0.0001')
plt.plot(trainLoss[2], label = 'learning rate 0.00001')
plt.xlabel('num of epochs')
plt.ylabel('training loss')
plt.legend()
plt.show()