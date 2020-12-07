""" The goal is to predict a linear function y = 3x+2 """
import random
from random import randrange
import numpy as np
import matplotlib.pyplot as plt

# create a dataset
dataset = []
for i in range(50):
    x = randrange(-25, 25)
    y = 3*x+20
    dataset.append((x, y))
# now split a dataset 80% in train and 20% in val
random.shuffle(dataset)
train = dataset[:40]
val = dataset[40:]
w = np.zeros(2) 
for i in train:
    y = np.dot(w, np.array([i[0], 1]))
    if np.sign(y) != np.sign(i[1]): 
        w = w + np.array([i[0], 1]) * i[1]
corrects = 0

for i in val: 
    y = np.dot(w, np.array([i[0], 1]))
    if (3*i[0]+20) > 0:
        plt.scatter(i[0], i[1], color='b')
    else:
        plt.scatter(i[0], i[1], color='r')

    if(np.sign(y) == np.sign(i[1])):
        corrects += 1
# plot a function you got
space = np.linspace(-5, 5, 10)
result_y = w[0]*space + w[1]
plt.plot(space, result_y, 'r')
plt.show()
print(f"Accuracy is {corrects/len(val)}")
print(w)
    
