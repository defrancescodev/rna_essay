import numpy as np
import matplotlib.pyplot as plt
from models.perceptron import Perceptron

X = np.array([
    [1, 1],
    [0, 1],
    [0, 2],
    [1, 0],
    [2, 2],
    [4, 1.5],
    [1.5, 6],
    [3, 5],
    [3, 3],
    [6, 4],
])

d = np.array([
    [1],
    [-1],
    [1],
    [1],
    [1],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
])


modelo = Perceptron()
modelo.fit(X, d)


fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[:5, 0], X[:5, 1])
ax.scatter(X[5:, 0], X[5:, 1])
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)

x1 = np.linspace(-4, 10)


w_inicial = modelo.history_w[0]
x2 = -w_inicial[1, 0] / w_inicial[2, 0] * x1 + w_inicial[0, 0] / w_inicial[2, 0]
x2 = np.nan_to_num(x2)
plt.plot(x1, x2, c='k')


linha = None
for w_epoca in modelo.history_w[1:]:
    x2 = -w_epoca[1, 0] / w_epoca[2, 0] * x1 + w_epoca[0, 0] / w_epoca[2, 0]
    x2 = np.nan_to_num(x2)
    if linha is not None:
        linha[0].remove()
    linha = ax.plot(x1, x2, c='k')
    plt.pause(0.1)


w_final = modelo.w
x2 = -w_final[1, 0] / w_final[2, 0] * x1 + w_final[0, 0] / w_final[2, 0]
x2 = np.nan_to_num(x2)
linha = ax.plot(x1, x2, c='b', lw=3)

plt.title(f"Épocas: {modelo.epoca}")
plt.show()