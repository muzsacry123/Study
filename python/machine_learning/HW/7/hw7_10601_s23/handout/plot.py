import numpy as np
import matplotlib.pyplot as plt

val_avg_log_likelihood = []
train_avg_log_likelihood = []
with open("avg_log_likelihood.txt", 'r') as f:
    for line in f:
        val_avg_log_likelihood.append(float(line))
        
with open("train_avg_log_likelihood.txt", 'r') as f:
    for line in f:
        train_avg_log_likelihood.append(float(line))
        
print(train_avg_log_likelihood)

plt.plot(np.log([10,100,1e3,1e4]), train_avg_log_likelihood[:-2], label='Train')
plt.plot(np.log([10,100,1e3,1e4]), val_avg_log_likelihood[:-2], label='Validation')
plt.title('Avg Log-likelihood VS. Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Avg Log-likelihood')
plt.legend()
plt.show()
