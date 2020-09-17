import numpy as np
import random
import matplotlib.pyplot as plt
# assume that those are hours of studying and scores on the test
dataset = np.array([(0, 5), (1, 10), (2, 13), (5, 20), (7, 24), (9, 25),
                   (12, 17),  (15, 28), (20, 34), (32, 50), (43, 55),
                   (56, 61), (76, 67), (89, 69), (92, 70),
                   (100, 80), (120, 80),  (150, 88), (156, 90), (158, 94),
                   (160, 98), (170, 100), (200, 100), (300, 100), (400, 100)])

random.shuffle(dataset)
train = dataset[:20]
val = dataset[20:]

ar = np.array([train[:, 0], np.ones(20)])
ar = ar.T

print(ar)

X_dag = np.dot(np.linalg.inv(np.dot(ar.T, ar)), ar.T)
w = np.dot(X_dag, train[:, 1])
corrects = 0
for sample in val:
    y = np.dot(w, np.array([sample[0], 1]))
    plt.scatter(sample[0], sample[1])
    print(f"MY Y IS {y}")
    print(f"REAL Y IS {sample[1]}")

space = np.linspace(-150, 150, 1000)
result_y = w[0]*space+w[1]
plt.plot(space, result_y, 'r')
plt.show()
