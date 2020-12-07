import numpy as np
import random
import matplotlib.pyplot as plt


def logfcn(s):
    return np.exp(s)/(1+np.exp(s))


# assume that those are hours of studying and scores on the test
dataset = np.array([(0, 0), (1, 0), (2, 0), (5, 0), (7, 0), (9, 0),
                   (12, 0),  (15, 0), (20, 1), (32, 0), (43, 0),
                   (56, 0), (76, 1), (89, 1), (92, 1),
                   (100, 1), (120, 1),  (150, 0), (156, 1), (158, 1),
                   (160, 1), (170, 1), (200, 1), (300, 1), (400, 1)])


random.shuffle(dataset)
train = dataset[:20]
val = dataset[20:]
N = len(train)
w = np.zeros(2)
for t in range(20):
    x = train[:, 0]
    y = train[:, 1]
    joined_x = np.array([x, np.ones(len(x))])
    mul = (x*y)/(1 + np.exp(y*np.dot(w, np.array([x, np.ones(len(x))]))))
    grad = -1/N * np.sum(mul)
    v = -grad
    w = w + 0.01*v

corrects = 0
for sample in val:
    y = logfcn(np.dot(w, np.array([sample[0], 1])))
    plt.scatter(sample[0], sample[1])
    print(f"MY Y IS {y}")
    print(f"REAL Y IS {sample[1]}")

space = np.linspace(-150, 150, 1000)
s = w[0]*space+w[1]
# result_y = s
result_y = np.exp(s)/(1+np.exp(s))

plt.plot(space, result_y, 'r')
plt.show()
