import numpy as np
import math
from typing import Callable

a = np.array([[1,2,4]])
b = np.array([[0,2,3]])
c = np.array([[0,2,3], [2, 3, 1]])
d = np.array([[4,2,3], [2, 3, 4]])

class Sigmoid(object):
    def __init__(self):
        '''
        Initialize state for sigmoid activation layer
        '''
        # Create cache to hold values for backward pass
        self.cache: dict[str, np.ndarray] = dict()

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Take sigmoid of input x.
        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape (output_size,)
        '''
        # TODO: implement this!
        self.cache['input'] = x
        return 1/(1+np.exp(-x))
        
    
    def backward(self, dz: np.ndarray) -> np.ndarray:
        '''
        :param dz: partial derivative of loss with respect to output of sigmoid activation
        :return: partial derivative of loss with respect to input of sigmoid activation
        '''
        # TODO: implement this based on your written answers!
        return self.cache['input']*dz
    
    
s = Sigmoid()
result = s.forward(b)
inputX = s.cache

back = s.backward(a)


print(a.T@b)

print(np.argmax(c, axis = 1))
error = np.sum(np.argmax(c, axis = 1) != np.argmax(d, axis = 1))

label = np.zeros(c.shape)
for i in range(len(c)):
    label[i, np.argmax(c[i])] = 1
    
d = np.random.rand(3,2)